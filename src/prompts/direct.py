from __future__ import annotations
from typing import List
from llm.types import Message

SYSTEM_PROMPT = """\
You are an expert question-answering system.
Answer multi-hop questions by explicitly connecting the required facts.
Be concise and factual.
"""

USER_TEMPLATE = """\
Question:
{question}

Instructions:
- Identify the intermediate fact or entity needed to answer the question.
- Use it to derive the final answer.
- End with the final answer on a new line as:
Answer: <final answer>
"""

def build_messages(question: str) -> List[Message]:
    """
    Build system + user messages from a raw question.
    """
    user_prompt = USER_TEMPLATE.format(question=question)

    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]