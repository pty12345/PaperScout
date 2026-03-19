# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import re
import threading
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from arft.reward_loop import ARFTRewardLoopWorker as RewardLoopWorker
from recipe.paper_search.prompts import *
from recipe.paper_search.utils import (
    ArXivClient,
    Paper,
    PaperPool,
)
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_rule_title(raw_title, meta_string):
    if not meta_string and not raw_title:
        return None
    ### Rule 1
    if meta_string:
        m = re.search(r'[“"](.+?),["”]', meta_string)
    else:
        m = None
    rule_title = m.group(1) if m else None
    if rule_title:
        return rule_title

    ## Rule 2
    if meta_string:
        m = re.search(r"\.:\s*(.+?)\.", meta_string, flags=re.S)  # re.S = DOTALL
    else:
        m = None
    rule_title = m.group(1) if m else None
    if rule_title:
        return rule_title

    ### Rule 3
    rule_title = re.sub(r",\s*\d{4}\.?\s*$", "", raw_title)
    if rule_title:
        return rule_title


@register("paper_search_agent")
class PaperSearchAgentFlow(AgentFlowBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, reward_loop_worker, tokenizer, processor, **kwargs)
        self.max_steps = kwargs.get("max_steps", 5)
        self.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        self.reward_top_k = kwargs.get("reward_top_k", 3)
        self.reward_threshold = kwargs.get("score_threshold", 0.4)
        self.search_cost = kwargs.get("search_cost", 0.1)
        self.expand_cost = kwargs.get("expand_cost", 0.05)
        self.use_discrete_reward = kwargs.get("use_discrete_reward", True)

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = PAPERSEARCH_TOOL_SCHEMAS
        self.client = ArXivClient(timeout=30.0)

        self.paper_pool = PaperPool()
        # Protect paper_pool read/write across concurrent search/expand tasks in the same event loop.
        # Keep the critical sections small (in-memory ops only).
        self._paper_pool_lock = threading.RLock()
        self.history_search_queries = {}
        self.user_query = ""
        self.steps = []

        self.history_actions = []

    @staticmethod
    def _normalize_arxiv_id(arxiv_id: str) -> str:
        """
        Normalize arXiv id by stripping version suffix.
        Example: '2401.12345v2' -> '2401.12345'
        """
        if not arxiv_id:
            return arxiv_id
        m = re.match(r"^(.*)v\d+$", arxiv_id)
        return m.group(1) if m else arxiv_id

    @staticmethod
    def _copy_paper_with_arxiv_id(paper: Paper, arxiv_id: str) -> Paper:
        """
        Return a new Paper instance with updated arxiv_id (pydantic v1/v2 compatible).
        """
        try:
            # pydantic v2
            return paper.model_copy(update={"arxiv_id": arxiv_id})
        except AttributeError:
            # pydantic v1
            return paper.copy(update={"arxiv_id": arxiv_id})

    def _format_history_actions(self) -> str:
        """
        根据历史 actions（用 self.history_actions 而不是 self.history_search_queries）格式化历史记录。
        输出与 refcode 类似格式:
        [Success] <action string>
        [Fail] <action string>
        """
        if not self.history_actions:
            return "None"

        lines: list[str] = []
        for action_pair in self.history_actions:
            # 假设每个 action 是字典，包含 'query'（或'action_str'）和 'return_num' 字段
            action, operation_result = action_pair
            if action == "search":
                query = operation_result
                # success_num = self._history_search_query_returns.get(query)
                # success_flag = '(Fail)' if (success_num is None or success_num == 0 ) else '(Success)'
                # lines.append(f"[Search] {success_flag} {query}")
                lines.append(f"[Search] {query}")
            elif action == "expand":
                arxiv_id = operation_result
                lines.append(f"[Expand] {arxiv_id}")
            else:
                raise ValueError(f"Invalid action: {action}")

        return "\n".join(lines) if lines else "None"

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        query = raw_prompt[0]["content"]
        self.user_query = query

        metrics = {}
        total_search_action_count = 0
        total_expand_action_count = 0

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1

            messages = [
                {"role": "system", "content": PAPERSEARCH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PAPERSEARCH_USER_PROMPT.format(
                        user_query=self.user_query,
                        paper_list=self.paper_pool.paper_list,
                        history_actions=self._format_history_actions(),
                    ),
                },
            ]

            prompt_ids = await self.apply_chat_template(
                messages,
                tools=self.tool_schemas,
            )

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

            if not tool_calls:
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=0.0,
                    extra_fields={
                        # NOTE:
                        # Put counters into reward_extra_info so they can be extracted into
                        # batch.non_tensor_batch and then show up in reward_extra_infos_dict
                        # (used by dumping/validation aggregation in the trainer).
                        "reward_extra_info": {
                            "search_actions_total": total_search_action_count,
                            "expand_actions_total": total_expand_action_count,
                        },
                    },
                )
                step = await self._postprocess(step, **kwargs)
                self.steps.append(step)
                break

            tool_calls = tool_calls[: self.max_parallel_calls]

            tasks = []
            for tool_call in tool_calls:
                try:
                    tool_args = json.loads(tool_call.arguments)
                    if tool_call.name == "search":
                        query = tool_args.get("query")
                        if query:
                            tasks.append(self.search(query, **kwargs))
                            self.history_actions.append(("search", query))
                            total_search_action_count += 1
                    elif tool_call.name == "expand":
                        arxiv_id = tool_args.get("arxiv_id")
                        if arxiv_id:
                            tasks.append(self.expand(arxiv_id, sampling_params, **kwargs))
                            self.history_actions.append(("expand", arxiv_id))
                            total_expand_action_count += 1
                except Exception as e:
                    print(f"Error in tool call: {e}")
                    continue

            with simple_timer("tool_calls", metrics):
                reward_scores = await asyncio.gather(*tasks)

            reward_score = sum(reward_scores)

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=reward_score,
                extra_fields={
                    "reward_extra_info": {
                        "search_actions_total": total_search_action_count,
                        "expand_actions_total": total_expand_action_count,
                    },
                },
            )
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

        return AgentFlowOutput(steps=self.steps, metrics=metrics)

    async def search(self, query: str, top_k: int = 10, **kwargs) -> float:
        if query in self.history_search_queries:
            return -0.5

        try:
            papers = await self.client.search(query, top_k, 'id <= "2404"')
        except Exception as e:
            print(f"Error in search {query}: str={str(e)}, type={type(e).__name__}, repr={repr(e)}")
            self.history_search_queries[query] = 0
            return 0.0
        new_papers = []
        tasks = []
        seen_base_ids: set[str] = set()
        for paper in papers:
            base_id = self._normalize_arxiv_id(paper.arxiv_id)
            # Dedup within this batch (e.g., v1/v2 variants or repeated results).
            if base_id in seen_base_ids:
                continue
            seen_base_ids.add(base_id)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(base_id):
                    continue

            paper_norm = self._copy_paper_with_arxiv_id(paper, base_id)
            new_papers.append(paper_norm)
            tasks.append(self.get_relevance_score(self.user_query, paper_norm, **kwargs))
        relevance_scores = await asyncio.gather(*tasks)

        keeped_papers, keeped_scores = [], []
        for paper, score in zip(new_papers, relevance_scores):
            if score < 0.01:
                continue

            # Re-check under lock to make has+add atomic across concurrent tasks.
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.arxiv_id):
                    continue

                keeped_papers.append(paper)
                keeped_scores.append(score)
                self.paper_pool.add_paper(paper, "search", query, score)

        self.history_search_queries[query] = len(keeped_papers)

        if len(keeped_papers) == 0:
            return 0.0

        sorted_scores = sorted(keeped_scores, reverse=True)
        top_k_scores = sorted_scores[: self.reward_top_k]
        if self.use_discrete_reward:
            top_k_scores = [1.0 if score >= self.reward_threshold else 0.0 for score in top_k_scores]
        reward_score = sum(top_k_scores) - self.search_cost
        return reward_score

    async def expand(self, arxiv_id: str, sampling_params: dict[str, Any], **kwargs) -> float:
        base_id = self._normalize_arxiv_id(arxiv_id)
        # Atomically: get entry + check expand + mark expand=True.
        with self._paper_pool_lock:
            paper_pool_entry = self.paper_pool.get_paper(base_id)
            if not paper_pool_entry:
                return -0.5
            if paper_pool_entry.expand:
                return -0.5
            paper_pool_entry.expand = True
        try:
            fulltext = await self.client.get_fulltext(base_id, download=True)
        except Exception as e:
            print(f"Error in get_fulltext {arxiv_id}: str={str(e)}, type={type(e).__name__}, repr={repr(e)}")
            return 0.0

        if fulltext is None:
            print(f"Fulltext is None for {arxiv_id}")
            return 0.0

        # STEP 2: extract reference titles
        ref_titles: list[str] = []
        references = fulltext.get("references") or {}
        for citation in references.keys():
            raw_title = references[citation].get("title")
            meta_string = references[citation].get("meta_string")

            rule_title = extract_rule_title(raw_title, meta_string)
            if rule_title:
                ref_titles.append(rule_title)

        # 去重（保持顺序），减少重复 lookup
        seen_titles: set[str] = set()
        ref_titles = [t for t in ref_titles if not (t in seen_titles or seen_titles.add(t))]

        async def _safe_lookup(title: str):
            try:
                return await self.client.lookup_by_title(title)
            except Exception as e:
                print(
                    f"Error in lookup_by_title title={title!r}: str={str(e)}, type={type(e).__name__}, repr={repr(e)}"
                )
                return []

        # 并行 lookup（lookup_by_title 默认不受全局 semaphore 限制，避免被慢接口拖住）
        lookup_results = await asyncio.gather(*(_safe_lookup(t) for t in ref_titles))
        expand_papers = [p for papers in lookup_results for p in papers]

        new_papers = []
        tasks = []
        seen_base_ids: set[str] = set()
        for paper in expand_papers:
            base_pid = self._normalize_arxiv_id(paper.arxiv_id)
            if base_pid in seen_base_ids:
                continue
            seen_base_ids.add(base_pid)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(base_pid):
                    continue
            paper_norm = self._copy_paper_with_arxiv_id(paper, base_pid)
            new_papers.append(paper_norm)
            tasks.append(self.get_relevance_score(self.user_query, paper_norm, **kwargs))
        relevance_scores = await asyncio.gather(*tasks)

        keeped_papers, keeped_scores = [], []
        for paper, score in zip(new_papers, relevance_scores):
            if score < 0.01:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.arxiv_id):
                    continue

                keeped_papers.append(paper)
                keeped_scores.append(score)
                self.paper_pool.add_paper(paper, "expand", paper_pool_entry.paper.title, score)

        if len(keeped_papers) == 0:
            return 0.0

        sorted_scores = sorted(keeped_scores, reverse=True)
        top_k_scores = sorted_scores[: self.reward_top_k]
        if self.use_discrete_reward:
            top_k_scores = [1.0 if score >= self.reward_threshold else 0.0 for score in top_k_scores]
        reward_score = sum(top_k_scores) - self.expand_cost
        return reward_score

    async def get_relevance_score(self, query: str, paper: Paper, **kwargs) -> float:
        prompt = SELECT_PROMPT.format(title=paper.title, abstract=paper.abstract, user_query=query)
        # Use ARFT reward loop worker (DisRM) instead of an externally managed /classify service.
        #
        # Requirements:
        # - reward_model.enable=True
        # - reward_model.rollout.name configured (vllm/sglang)
        # - reward_model.model.path points to the selector/DisRM model
        try:
            result = await self.reward_loop_worker.compute_score_disrm.remote(prompt)
        except Exception as e:
            raise RuntimeError(
                "PaperSearchAgentFlow relevance scoring now uses RewardLoopWorker (DisRM) instead of external "
                "/classify endpoints. Please ensure reward_model.enable=True and reward_model.model.path is set "
                "to your selector/DisRM model."
            ) from e

        return float(1.0 - result["reward_score"])
