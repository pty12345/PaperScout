import asyncio
import pickle
from functools import total_ordering
from typing import Any, Optional

import httpx
from pydantic import BaseModel
from sortedcontainers import SortedList

from .http_retry import httpx_request_with_retry

ZIP_INDEX_DICT = {
    "warning": pickle.load(open("/data/tingyue/workspace/oyj/arxiv_database/id2paper_warning.pkl", "rb")),
    "no_problem": pickle.load(open("/data/tingyue/workspace/oyj/arxiv_database/id2paper_no_problem.pkl", "rb")),
}


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    categories: str = ""
    authors: str = ""
    update_date: str = ""
    score: float = 0.0  # 检索分数，不用于排序


class QueueStats(BaseModel):
    queue_size: int
    max_queue_size: int
    batch_size: int
    batch_timeout: float
    # BatchProcessor runtime metrics
    running: bool
    task_done: bool
    last_exception: Optional[str] = None
    submitted_total: int
    dequeued_total: int
    processed_total: int
    batches_total: int
    last_batch_size: int
    last_process_age_sec: Optional[float] = None


@total_ordering
class PaperPoolEntry(BaseModel):
    paper: Paper
    source: str  # 'search' 或 'expand'
    origin: str  # 如果 source='search'，origin 表示 query；如果 source='expand'，origin 表示父 paper 的 title
    score: float  # 论文的分数，用于排序
    expand: bool = False  # 是否已经扩展
    exist_local: bool = False  # 是否存在本地

    def __lt__(self, other):
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        if self.score != other.score:
            return self.score < other.score
        return self.paper.arxiv_id < other.paper.arxiv_id

    def __eq__(self, other):
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        return self.score == other.score and self.paper.arxiv_id == other.paper.arxiv_id

    def __hash__(self):
        return hash(self.paper.arxiv_id)


class ArXivClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float = 30.0,
        *,
        max_concurrency: Optional[int] = 16,
        max_fulltext_concurrency: Optional[int] = 8,
    ):
        self.base_url = base_url.rstrip("/")

        # 使用持久化 session
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0, pool=60.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        )

        self._semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None
        )
        # 独立的信号量用于控制 get_fulltext 的并发，避免服务端过载
        self._fulltext_semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_fulltext_concurrency)
            if max_fulltext_concurrency and max_fulltext_concurrency > 0
            else None
        )

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return await httpx_request_with_retry(self.client, method, url, semaphore=self._semaphore, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """关闭底层 HTTP 连接"""
        await self.client.aclose()

    async def search(self, query: str, top_k: int = 10, filter_expr: Optional[str] = None) -> list[Paper]:
        """语义搜索"""
        params = {"q": query, "top_k": top_k}
        if filter_expr:
            params["filter"] = filter_expr

        resp = await self._request("GET", "/search", params=params)
        resp.raise_for_status()

        # Pydantic 自动解析列表中的字典
        return [Paper(**item) for item in resp.json()]

    async def query(self, filter_expr: str, top_k: int = 10) -> list[Paper]:
        """条件/关键字查询 (非向量搜索)"""
        params = {"filter": filter_expr, "top_k": top_k}
        resp = await self._request("GET", "/query", params=params)
        resp.raise_for_status()
        return [Paper(**item) for item in resp.json()]

    async def lookup_by_title(self, title: str) -> list[Paper]:
        """根据标题精确查找"""
        params = {"title": title}
        resp = await self._request("GET", "/lookup", params=params)
        resp.raise_for_status()
        return [Paper(**item) for item in resp.json()]

    @staticmethod
    def _record_to_paper(data: dict[str, Any]) -> Paper:
        """
        Convert server record (Milvus raw dict) into Paper.

        Server may return `id` instead of `arxiv_id` for /paper/{arxiv_id}.
        """
        # Prefer explicit arxiv_id, fallback to id.
        arxiv_id = data.get("arxiv_id") or data.get("id") or ""
        return Paper(
            arxiv_id=arxiv_id,
            title=data.get("title", "") or "",
            abstract=data.get("abstract", "") or "",
            categories=data.get("categories", "") or "",
            authors=data.get("authors", "") or "",
            update_date=data.get("update_date", "") or "",
            score=float(data.get("score", 0.0) or 0.0),
        )

    async def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """获取单篇论文详情"""
        resp = await self._request("GET", f"/paper/{arxiv_id}")
        resp.raise_for_status()
        data = resp.json()
        return self._record_to_paper(data) if data else None

    async def get_fulltext(self, arxiv_id: str, download: bool = False) -> Optional[dict[str, Any]]:
        """获取论文全文 (HTML JSON 结构)

        使用独立的信号量控制并发，避免一次性请求过多导致服务端崩溃。
        """
        resp = await httpx_request_with_retry(
            self.client,
            "GET",
            f"/paper/{arxiv_id}/fulltext",
            params={"download": download},
            semaphore=self._fulltext_semaphore,
        )
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            error_body = resp.json()
            raise RuntimeError(error_body.get("detail", "Unknown server error")) from e

    async def health(self) -> dict:
        """健康检查"""
        resp = await self._request("GET", "/health")
        return resp.json()

    async def queue_stats(self) -> QueueStats:
        """查看批处理队列状态"""
        resp = await self._request("GET", "/queue")
        resp.raise_for_status()
        return QueueStats(**resp.json())


class PaperPool:
    def __init__(self, max_size: int = 20, threshold: float = 0.0, max_abstract_words: int = 400):
        self.papers = {}  # arxiv_id -> PaperPoolEntry
        self.ranked_papers = SortedList()
        self.max_size = max_size
        self.threshold = threshold
        self.max_abstract_words = max_abstract_words

        self.exist_local_papers = set()
        for zip_name, zip_index in ZIP_INDEX_DICT.items():
            self.exist_local_papers.update(zip_index.keys())

    def add_paper(self, paper: Paper, source: str, origin: str, score: float):
        if paper.arxiv_id in self.papers:
            return

        arxiv_id = paper.arxiv_id.split("v")[0]
        exist_local = True if arxiv_id in self.exist_local_papers else False

        paper_pool_entry = PaperPoolEntry(
            paper=paper, source=source, origin=origin, score=score, exist_local=exist_local
        )
        self.papers[paper.arxiv_id] = paper_pool_entry
        self.ranked_papers.add(paper_pool_entry)

    def get_paper(self, arxiv_id: str) -> Optional[PaperPoolEntry]:
        return self.papers.get(arxiv_id)

    def has_paper(self, arxiv_id: str) -> bool:
        return arxiv_id in self.papers

    @property
    def paper_list(self) -> str:
        """
        Format the paper list into a string.
        The list includes up to max_size/2 expanded papers ([EXP]) and max_size/2 unexpanded papers ([NEW]).
        Abstracts are truncated to a maximum length.
        """
        if not self.papers:
            return "No papers in the pool."

        # 分离已扩展和未扩展的论文，并按分数从高到低排序
        expanded_entries = [e for e in self.ranked_papers if e.expand and e.score >= self.threshold and e.exist_local]
        unexpanded_entries = [
            e for e in self.ranked_papers if not e.expand and e.score >= self.threshold and e.exist_local
        ]

        # SortedList 是升序的，所以我们需要 reverse
        expanded_entries.reverse()
        unexpanded_entries.reverse()

        half_size = self.max_size // 2
        top_expanded = expanded_entries[:half_size]
        top_unexpanded = unexpanded_entries[:half_size]

        # 合并并再次按分数排序
        display_entries = top_expanded + top_unexpanded
        display_entries.sort(key=lambda x: x.score, reverse=True)

        if not display_entries:
            return "No relevant papers found above threshold."

        description = (
            "Paper Pool Status:\n"
            "- [EXP]: Paper has been expanded (already used as a seed for more papers).\n"
            "- [NEW]: New paper found via search or expansion, candidate for further exploration.\n"
            "- Format: [arxiv_id] (score) [STATUS] Title\n"
        )

        lines = [description]
        for entry in display_entries:
            paper = entry.paper
            status_tag = "[EXP]" if entry.expand else "[NEW]"

            # 截断摘要
            abstract = paper.abstract
            words = abstract.split()
            if len(words) > self.max_abstract_words:
                abstract = " ".join(words[: self.max_abstract_words]) + "..."

            entry_str = f"[{paper.arxiv_id}] ({entry.score:.2f}) {status_tag} {paper.title}\nAbstract: {abstract}"
            lines.append(entry_str)

        return "\n\n".join(lines)


def parse_full_paper_to_specific_sections(d, key_words: list, result=None):
    """
    递归检索所有key为'title'，且value包含'Introduction'或'Related Work'关键字的章节信息。
    d: 可能为dict, list, 或其他类型
    result: 结果列表，元素为({父dict/section}, title字符串)
    """
    if result is None:
        result = []
    if isinstance(d, dict):
        title = d.get("title")
        if isinstance(title, str):
            # 检查是否包含任意关键字
            for kw in key_words:
                if kw.lower() in title.lower():
                    result.append(d)
                    break
        for v in d.values():
            parse_full_paper_to_specific_sections(v, key_words, result)
    elif isinstance(d, list):
        for item in d:
            parse_full_paper_to_specific_sections(item, key_words, result)
    return result
