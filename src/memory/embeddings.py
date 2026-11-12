"""The fixed DashScope embedding model and validation of its dense output."""

import math
import os
from numbers import Real

from dashscope import TextEmbedding

EMBEDDING_MODEL = "qwen3.7-text-embedding"


class EmbeddingError(ValueError):
    """An embedding could not be obtained or is not safe to persist."""


def validate_vector(vector: object, dimension: int | None = None) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise EmbeddingError("向量必须是非空数值列表")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in vector):
        raise EmbeddingError("向量只能包含数值，不能包含布尔值或字符串")
    values = [float(value) for value in vector]
    # Milvus FLOAT_VECTOR stores IEEE-754 float32 values.
    if any(
        not math.isfinite(value) or abs(value) > 3.4028234663852886e38
        for value in values
    ):
        raise EmbeddingError("向量必须包含可由 FLOAT_VECTOR 表示的有限数值")
    if dimension is not None and len(values) != dimension:
        raise EmbeddingError(
            f"向量维度不兼容：集合为 {dimension}，返回向量为 {len(values)}"
        )
    return values


class DashScopeEmbedder:
    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("向量化文本不能为空")
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise EmbeddingError("缺少环境变量 DASHSCOPE_API_KEY")
        try:
            response = TextEmbedding.call(
                model=EMBEDDING_MODEL,
                input=[text],
                api_key=api_key,
                text_type="document",
                request_timeout=60,
            )
        except Exception as exc:
            detail = str(exc).replace(api_key, "[redacted]")
            raise EmbeddingError(
                f"DashScope 向量化调用失败：{type(exc).__name__}: {detail}"
            ) from exc
        if response.status_code != 200:
            detail = str(response.message).replace(api_key, "[redacted]")
            raise EmbeddingError(
                f"DashScope {EMBEDDING_MODEL} 返回 {response.status_code} "
                f"({response.code}): {detail}"
            )
        try:
            embeddings = response.output["embeddings"]
            if len(embeddings) != 1 or embeddings[0]["text_index"] != 0:
                raise EmbeddingError("向量化输出与输入文本的对应关系无效")
            return validate_vector(embeddings[0]["embedding"])
        except (KeyError, TypeError, IndexError) as exc:
            raise EmbeddingError("DashScope 返回的向量结构不合法") from exc
