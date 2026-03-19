PAPERSEARCH_SYSTEM_PROMPT = "You are a research agent. Your goal is to find papers relevant to the User Query."

PAPERSEARCH_USER_PROMPT = """### User Query 
{user_query}

### History Actions
{history_actions}

### Paper List
{paper_list}

### Instructions
Analyze the **Paper List** and **History Actions** to determine the next set of actions. Enclose your analysis of the state and decision logic within `<analysis>...</analysis>` tags.
**You support parallel tool calling.** You should output multiple tool calls in a single step if several independent actions are valuable at the current state.
**Attend to the history actions and avoid expanding the same papers.**

### Output Format
<analysis>
[Your analysis of the current state and decision logic...]
</analysis>
<tool_call>
[Tool call 1]
</tool_call>
<tool_call>
[Tool call 2]
</tool_call>
...
"""

SELECT_PROMPT = """You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.

Searched Paper:
Title: {title}
Abstract: {abstract}

User Query: {user_query}

Output format: Decision: True/False
Reason:... 
Decision:"""

# OpenAI-compatible tool schemas for PaperSearch actions.
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": ("Search for relevant papers in the arXiv repository."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A single search query (natural language or keywords). "
                        "No field scopes (ti:/abs:) or boolean ops. Must differ from all history queries."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

EXPAND_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": (
            "Expand from an existing paper by following its references to surface additional relevant works. "
            "Use this when search is saturated and you want to broaden coverage around a known paper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": (
                        "The arXiv identifier (e.g., '1706.03762') of a paper already in the current paper list."
                    ),
                }
            },
            "required": ["arxiv_id"],
        },
    },
}

PAPERSEARCH_TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA, EXPAND_TOOL_SCHEMA]
