"""Main conversation agent; persistent memory is never a reply knowledge source."""

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

from memory.agents import MemoryAgent, MemoryResult
from memory.entities import EntityStore, ToolError
from memory.milvus import SemanticStore

CHAT_MODEL = "deepseek-flash"

MAIN_PROMPT = """你是与用户进行多轮对话的助手。仅根据本次真实聊天上下文回答。
你没有持久记忆召回能力，不声称知道以前会话中的内容。
系统会提供本轮记忆处理的执行状态，它只用于如实反馈执行情况，不是回答的知识来源。
completed 表示本轮所需操作已得到成功确认；skipped 表示没有写入；
partial 表示只有部分内容成功；failed 表示执行失败且没有确认成功的写入。
对于明确记忆请求，严格依据这个状态回答。partial/failed 时明确说明未完全记住，
skipped 时不能声称已保存；不要将“收到/理解”描述为“已经记住”。
舍弃证据不足的候选是正常决定，不是存储错误；failed_steps 为空时不虚构“确认环节失败”。
若有写入结果未知，说明无法确认，不能建议用户立即重复整批保存；重复追加可能产生重复记录。
记忆处理已在本轮自动执行，不要把已执行的保存描述成“需要你先确认是否保存”。
不透露内部工具消息、实体查询内容或模型分析。保持回复自然、简洁。
"""


def create_chat_model() -> ChatDeepSeek:
    """Use system credentials and the model fixed by the specification."""
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("缺少环境变量 DEEPSEEK_API_KEY")
    return ChatDeepSeek(model=CHAT_MODEL, temperature=0, timeout=45, max_retries=0)


def model_error(error: Exception) -> str:
    """Report the failure without exposing response bodies or credentials."""
    status = getattr(error, "status_code", None)
    suffix = f"，HTTP {status}" if isinstance(status, int) else ""
    return f"{type(error).__name__}{suffix}；请检查凭据、模型权限、网络及上下文长度"


def close_chat_model(model: Any) -> None:
    """Close both transports created by ChatDeepSeek in this synchronous CLI."""
    try:
        client = getattr(model, "root_client", None)
        if client is not None:
            client.close()
    finally:
        async_client = getattr(model, "root_async_client", None)
        if async_client is not None:
            asyncio.run(async_client.close())


class TurnResult(BaseModel):
    reply: str
    memory: MemoryResult
    answer_error: str | None = None


class Conversation:
    """A single in-memory conversation with injectable models and stores."""

    def __init__(
        self,
        memory_agent: MemoryAgent,
        model: Any,
        close_resources: Callable[[], None] | None = None,
    ) -> None:
        self.memory_agent = memory_agent
        self.model = model
        self.messages: list[BaseMessage] = []
        self._close_resources = close_resources

    def respond(self, text: str) -> TurnResult | None:
        if not text.strip():
            return None
        self.messages.append(HumanMessage(content=text))
        try:
            memory = self.memory_agent.run(list(self.messages))
        except Exception as error:
            memory = MemoryResult(
                status="failed",
                reasons=["本轮记忆处理异常结束"],
                errors=[ToolError(step="memory_agent", message=model_error(error))],
            )

        # Only counts and status cross the memory/reply boundary, never entities,
        # semantic text, query results, model reasoning or internal tool messages.
        execution = {
            "status": memory.status,
            "saved_entities": len(memory.entity_ids),
            "saved_semantics": len(memory.semantic_ids),
            "failed_steps": sorted({error.step for error in memory.errors}),
            "unknown_write": any(error.unknown for error in memory.errors),
            "discarded_candidates": len(memory.discarded),
            "unfinished_operations": len(memory.unfinished),
        }
        try:
            answer = self.model.invoke(
                [
                    SystemMessage(content=MAIN_PROMPT),
                    SystemMessage(
                        content="本轮记忆执行状态："
                        + json.dumps(execution, ensure_ascii=False)
                    ),
                    *self.messages,
                ]
            )
            if (
                not isinstance(answer, AIMessage)
                or answer.tool_calls
                or answer.invalid_tool_calls
                or not answer.text.strip()
            ):
                raise ValueError("主 Agent 未返回有效文本回复")
            reply = answer.text
            # Store only the public reply, not model metadata or tool state.
            self.messages.append(AIMessage(content=reply))
            return TurnResult(reply=reply, memory=memory)
        except Exception as error:
            detail = model_error(error)
            # No fabricated assistant message is inserted when answering fails.
            return TurnResult(
                reply="回答失败：" + detail, memory=memory, answer_error=detail
            )

    def close(self) -> None:
        close, self._close_resources = self._close_resources, None
        if close is not None:
            close()


def create_conversation(root: str | Path = ".memory") -> Conversation:
    entities = EntityStore(root)
    entities.initialize()
    semantics = SemanticStore(entities)
    model = None
    try:
        model = create_chat_model()
        memory = MemoryAgent(entities, semantics, model=model)
    except BaseException:
        try:
            semantics.close()
        finally:
            if model is not None:
                close_chat_model(model)
        raise

    def close_resources() -> None:
        try:
            semantics.close()
        finally:
            close_chat_model(model)

    return Conversation(memory, model, close_resources)
