"""Append-only semantic storage in the existing local Milvus database."""

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import DataNotMatchException, ParamError

from memory.embeddings import DashScopeEmbedder, validate_vector
from memory.entities import (
    EntityStore,
    ToolError,
    ToolResult,
    ToolSuccess,
    validate_entity_id,
)

COLLECTION_NAME = "memory"
TEXT_MAX_BYTES = 8000
ENTITY_CAPACITY = 1024
ENTITY_MAX_LENGTH = 36
ENTITY_MARKER = re.compile(r"\[\[([^\[\]]+)\]\]")


class CollectionCompatibilityError(ValueError):
    """The existing database must be preserved and cannot be used as-is."""


class SemanticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: StrictStr = Field(
        description=(
            "有充分证据、可独立理解的完整事实文本，最多 8000 UTF-8 字节。"
            "每个已确定实体必须写成 [[真实实体ID]]，不能只写姓名；无实体的明确记忆保留指定原文。"
        )
    )
    entities: list[StrictStr] = Field(
        description=(
            "必须与 text 中 [[真实实体ID]] 标记的去重集合一致，按首次出现顺序列出；"
            "不能只填此列表而在 text 中省略标记。无实体时为 []。"
        )
    )

    @model_validator(mode="after")
    def validate_record(self) -> "SemanticRecord":
        if not self.text.strip():
            raise ValueError("语义文本不能为空")
        if len(self.text.encode("utf-8")) > TEXT_MAX_BYTES:
            raise ValueError(
                "语义文本超过 8000 UTF-8 字节；请按完整事实拆分后重新提交，不能截断"
            )
        references = list(dict.fromkeys(ENTITY_MARKER.findall(self.text)))
        remainder = ENTITY_MARKER.sub("", self.text)
        if "[[" in remainder or "]]" in remainder:
            raise ValueError("实体引用标记必须完整，格式为 [[实体id]]")
        for entity_id in [*references, *self.entities]:
            validate_entity_id(entity_id)
            if len(entity_id) > ENTITY_MAX_LENGTH:
                raise ValueError("实体 ID 超过 36 字符")
        if set(references) != set(self.entities):
            raise ValueError("entities 必须与文本中的实体标记去重集合一致")
        if len(references) > ENTITY_CAPACITY:
            raise ValueError("关联实体超过 1024 个；请按完整事实拆分并重新计算引用")
        self.entities = references
        return self


class SaveSemanticsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    records: list[SemanticRecord] = Field(min_length=1)


class _LocalMilvusClient(MilvusClient):
    """Also release the embedded server; PyMilvus.close only closes its channel."""

    def __init__(self, path: Path):
        self._memory_path = path
        try:
            super().__init__(uri=str(path))
        except BaseException:
            self._release_server()
            raise

    def _release_server(self) -> None:
        from milvus_lite import server_manager_instance

        server_manager_instance.release_server(str(self._memory_path))

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._release_server()


def _database_path(root: str | Path) -> Path:
    root = Path(root)
    if root.is_symlink():
        raise ValueError("记忆目录不得是符号链接")
    path = root / "memory.db"
    if path.is_symlink():
        raise ValueError("拒绝符号链接形式的 memory.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def collection_dimension(client: MilvusClient, dimension: int | None = None) -> int:
    """Validate all persisted schema parameters without changing the collection."""
    schema = client.describe_collection(COLLECTION_NAME)
    fields = {field["name"]: field for field in schema.get("fields", [])}
    differences = []
    expected_types = {
        "id": DataType.INT64,
        "text": DataType.VARCHAR,
        "vector": DataType.FLOAT_VECTOR,
        "entities": DataType.ARRAY,
    }
    if set(fields) != set(expected_types):
        differences.append(f"字段应为 {list(expected_types)}，实际为 {list(fields)}")
    for name, expected in expected_types.items():
        field = fields.get(name, {})
        if field.get("type") != expected:
            differences.append(
                f"{name} 类型应为 {expected.name}，实际为 {field.get('type')}"
            )
        if field.get("nullable", False):
            differences.append(f"{name} 不能为 nullable")
        if name != "id" and field.get("is_primary", False):
            differences.append(f"{name} 不能为主键")
    if not schema.get("auto_id") or not fields.get("id", {}).get("is_primary"):
        differences.append("id 必须为自动生成的主键")
    if not fields.get("id", {}).get("auto_id"):
        differences.append("id.auto_id 必须为 true")
    if schema.get("enable_dynamic_field", False):
        differences.append("不能启用动态字段")
    text_params = fields.get("text", {}).get("params", {})
    if text_params.get("max_length") != TEXT_MAX_BYTES:
        differences.append(
            f"text.max_length 应为 {TEXT_MAX_BYTES}，实际为 {text_params.get('max_length')}"
        )
    entity_field = fields.get("entities", {})
    if entity_field.get("element_type") != DataType.VARCHAR:
        differences.append("entities.element_type 应为 VARCHAR")
    entity_params = entity_field.get("params", {})
    for name, expected in (
        ("max_capacity", ENTITY_CAPACITY),
        ("max_length", ENTITY_MAX_LENGTH),
    ):
        if entity_params.get(name) != expected:
            differences.append(
                f"entities.{name} 应为 {expected}，实际为 {entity_params.get(name)}"
            )
    actual_dimension = fields.get("vector", {}).get("params", {}).get("dim")
    if type(actual_dimension) is not int or actual_dimension <= 0:
        differences.append(f"vector.dim 无效：{actual_dimension}")
    elif dimension is not None and dimension != actual_dimension:
        differences.append(f"vector.dim 应为 {dimension}，实际为 {actual_dimension}")
    if differences:
        raise CollectionCompatibilityError(
            "memory 集合不兼容：" + "；".join(differences)
        )
    return actual_dimension


def create_client(dimension: int, root: str | Path = ".memory") -> MilvusClient:
    """Initialize the same four-field collection, preserving existing data."""
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("向量维度必须为正整数")
    path = _database_path(root)
    # Lite 3.2.2's gRPC CreateCollection translator drops ARRAY.max_length.
    # Its public native schema API preserves it in the very same local database.
    # Only creation uses this API; existing collections are never rewritten.
    from milvus_lite import CollectionSchema, FieldSchema, MilvusLite
    from milvus_lite import DataType as LiteDataType

    with MilvusLite(str(path)) as database:
        if not database.has_collection(COLLECTION_NAME):
            collection = database.create_collection(
                COLLECTION_NAME,
                CollectionSchema(
                    fields=[
                        FieldSchema(
                            "id", LiteDataType.INT64, is_primary=True, auto_id=True
                        ),
                        FieldSchema(
                            "text", LiteDataType.VARCHAR, max_length=TEXT_MAX_BYTES
                        ),
                        FieldSchema("vector", LiteDataType.FLOAT_VECTOR, dim=dimension),
                        FieldSchema(
                            "entities",
                            LiteDataType.ARRAY,
                            element_type=LiteDataType.VARCHAR,
                            max_capacity=ENTITY_CAPACITY,
                            max_length=ENTITY_MAX_LENGTH,
                        ),
                    ],
                    enable_dynamic_field=False,
                ),
            )
            collection.create_index(
                "vector", {"index_type": "FLAT", "metric_type": "COSINE"}
            )
    client = _LocalMilvusClient(path)
    try:
        collection_dimension(client, dimension)
        return client
    except BaseException:
        client.close()
        raise


class SemanticStore:
    def __init__(self, entities: EntityStore, embedder=None):
        self.entities = entities
        self.root = entities.root
        self.embedder = embedder if embedder is not None else DashScopeEmbedder()
        self.client: MilvusClient | None = None
        self.dimension: int | None = None

    def _open_existing(self) -> None:
        if self.client is not None:
            return
        path = _database_path(self.root)
        if not path.exists():
            return
        client = _LocalMilvusClient(path)
        try:
            if not client.has_collection(COLLECTION_NAME):
                client.close()
                return
            self.dimension = collection_dimension(client)
            self.client = client
        except BaseException:
            client.close()
            raise

    def save_semantics(self, records: list[dict]) -> ToolResult:
        """Preflight the entire batch, then append and acknowledge each item."""
        try:
            request = SaveSemanticsInput.model_validate({"records": records})
        except Exception as exc:
            return ToolResult(
                status="failed",
                errors=[ToolError(step="validate_semantics", message=str(exc))],
            )
        for index, record in enumerate(request.records):
            try:
                for entity_id in record.entities:
                    self.entities.get(entity_id)
            except Exception as exc:
                return ToolResult(
                    status="failed",
                    errors=[
                        ToolError(
                            index=index,
                            step="validate_semantics",
                            message=str(exc),
                        )
                    ],
                )
        try:
            self._open_existing()
        except Exception as exc:
            return ToolResult(
                status="failed",
                errors=[ToolError(step="open_collection", message=str(exc))],
            )

        successes = []
        errors = []
        for index, record in enumerate(request.records):
            try:
                vector = validate_vector(
                    self.embedder.embed(record.text), self.dimension
                )
            except Exception as exc:
                errors.append(
                    ToolError(index=index, step="embed_semantic", message=str(exc))
                )
                continue
            if self.client is None:
                try:
                    self.client = create_client(len(vector), self.root)
                    self.dimension = len(vector)
                except Exception as exc:
                    errors.append(
                        ToolError(index=index, step="open_collection", message=str(exc))
                    )
                    continue
            try:
                # A timeout argument overrides retry_times in the pinned SDK.
                # Leave it unset and explicitly disable its transport retries.
                result = self.client.insert(
                    collection_name=COLLECTION_NAME,
                    data=[
                        {
                            "text": record.text,
                            "vector": vector,
                            "entities": record.entities,
                        }
                    ],
                    retry_times=0,
                )
            except (DataNotMatchException, ParamError) as exc:
                errors.append(
                    ToolError(index=index, step="insert_semantic", message=str(exc))
                )
                continue
            except Exception as exc:
                errors.append(
                    ToolError(
                        index=index,
                        step="insert_semantic",
                        unknown=True,
                        message=f"写入结果未知，未自动重试：{type(exc).__name__}: {exc}",
                    )
                )
                continue
            ids = result.get("ids", []) if isinstance(result, dict) else []
            if (
                not isinstance(result, dict)
                or type(result.get("insert_count")) is not int
                or result.get("insert_count") != 1
                or not isinstance(ids, Sequence)
                or len(ids) != 1
                or type(ids[0]) is not int
                or not -(2**63) <= ids[0] < 2**63
            ):
                errors.append(
                    ToolError(
                        index=index,
                        step="confirm_semantic",
                        unknown=True,
                        message="写入响应未确认单条记录的真实主键，结果未知，未自动重试",
                    )
                )
                continue
            successes.append(
                ToolSuccess(
                    index=index,
                    data={
                        "id": ids[0],
                        "text": record.text,
                        "entities": record.entities,
                    },
                )
            )
        status = (
            "partial" if successes and errors else "failed" if errors else "completed"
        )
        return ToolResult(status=status, successes=successes, errors=errors)

    def close(self) -> None:
        client, self.client = self.client, None
        self.dimension = None
        if client is not None:
            client.close()
