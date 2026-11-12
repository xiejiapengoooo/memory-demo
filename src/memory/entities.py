"""Validated entity files and the shared storage-tool result contract."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

REQUIRED_FIELDS = frozenset({"id", "keywords", "agent_cognition", "relations"})


def validate_entity_id(value: str) -> str:
    """Accept only the fixed agent ID or a canonical, lower-case UUID v4."""
    if not isinstance(value, str):
        raise ValueError("实体 ID 必须为字符串")
    if value == "agent":
        return value
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("实体 ID 必须是标准小写 UUID v4 或 agent") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("实体 ID 必须是标准小写 UUID v4 或 agent")
    return value


def split_values(value: str) -> list[str]:
    return list(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )


def _check_utf8(value: Any) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("字符串包含无法编码为 UTF-8 的字符") from exc
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_utf8(key)
            _check_utf8(item)
    elif isinstance(value, list):
        for item in value:
            _check_utf8(item)


def _relations(value: dict[str, str]) -> dict[str, str]:
    result = {}
    for name, targets in value.items():
        ids = split_values(targets)
        for entity_id in ids:
            validate_entity_id(entity_id)
        if ids:
            result[name] = ",".join(ids)
    return result


class ToolSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int
    data: dict[str, Any]


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int | None = None
    step: str
    message: str
    unknown: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["completed", "partial", "failed"]
    successes: list[ToolSuccess] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)


class Entity(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    __pydantic_extra__: dict[str, StrictStr] = Field(init=False)

    id: StrictStr
    keywords: StrictStr
    agent_cognition: StrictStr
    relations: dict[StrictStr, StrictStr]

    _id = field_validator("id")(validate_entity_id)

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, value: str) -> str:
        return ",".join(split_values(value))

    _normalize_relations = field_validator("relations")(_relations)

    @model_validator(mode="after")
    def valid_utf8(self) -> Entity:
        _check_utf8(self.model_dump())
        return self


class EntityInitial(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    __pydantic_extra__: dict[str, StrictStr] = Field(init=False)

    keywords: StrictStr = Field(
        default="",
        description="可查询的名称、别名或关键词，以英文逗号分隔；没有时为空字符串。",
    )
    agent_cognition: StrictStr = Field(
        default="",
        description="有充分依据的当前认知；尚无可靠认知时为空字符串。以后可通过 update_entities.attributes 更新。",
    )
    relations: dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        description="新实体创建时必须为空对象；获取真实 ID 后再建立关系。",
    )

    @model_validator(mode="after")
    def initial_only(self) -> EntityInitial:
        _check_utf8(self.model_dump())
        if "id" in (self.__pydantic_extra__ or {}):
            raise ValueError("创建实体时不得提供 ID；ID 由程序生成")
        if self.relations:
            raise ValueError("创建实体时 relations 必须为空；保存后再建立关系")
        self.keywords = ",".join(split_values(self.keywords))
        return self


class QueryEntitiesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ids: list[StrictStr] = Field(default_factory=list)
    keywords: list[StrictStr] = Field(default_factory=list)

    @field_validator("ids")
    @classmethod
    def valid_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(validate_entity_id(value) for value in values))

    @field_validator("keywords")
    @classmethod
    def valid_keywords(cls, values: list[str]) -> list[str]:
        _check_utf8(values)
        return list(
            dict.fromkeys(value.strip().casefold() for value in values if value.strip())
        )

    @model_validator(mode="after")
    def nonempty_query(self) -> QueryEntitiesInput:
        if not self.ids and not self.keywords:
            raise ValueError("查询至少需要一个有效 ID 或关键词")
        return self


class CreateEntitiesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    entities: list[EntityInitial] = Field(min_length=1)


class EntityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: StrictStr
    attributes: dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        description=(
            "需要设置或更新的字符串属性，可包含 keywords、agent_cognition 和 name 等扩展字段；"
            "不可包含 id 或 relations。必填字段可更新但不可删除。"
            "例如明确离职后，attributes={'agent_cognition':'曾在甲公司工作，现已离职。'}，"
            "同时按已知信息更新其他受影响属性，避免保留过时的当前任职认知。"
        ),
    )
    remove_attributes: list[StrictStr] = Field(default_factory=list)
    relations_add: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    relations_remove: dict[StrictStr, StrictStr] = Field(default_factory=dict)

    _id = field_validator("id")(validate_entity_id)
    _normalize_relations = field_validator("relations_add", "relations_remove")(
        _relations
    )

    @model_validator(mode="after")
    def no_conflicts(self) -> EntityUpdate:
        _check_utf8(self.model_dump())
        if {"id", "relations"} & self.attributes.keys():
            raise ValueError("attributes 不得包含 id 或 relations")
        if REQUIRED_FIELDS.intersection(self.remove_attributes):
            raise ValueError("不可移除实体必填字段")
        if self.attributes.keys() & set(self.remove_attributes):
            raise ValueError("同一属性不能同时设置和移除")
        for name in self.relations_add.keys() & self.relations_remove.keys():
            if set(split_values(self.relations_add[name])) & set(
                split_values(self.relations_remove[name])
            ):
                raise ValueError("同一关系目标不能同时添加和移除")
        return self


class UpdateEntitiesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    updates: list[EntityUpdate] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> UpdateEntitiesInput:
        ids = [update.id for update in self.updates]
        if len(ids) != len(set(ids)):
            raise ValueError("一批更新中同一实体只能出现一次")
        return self


class EntityDataError(ValueError):
    """Existing data is invalid; callers must not treat it as a missing entity."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EntityDataError(f"JSON 含有重复字段：{key}")
        result[key] = value
    return result


def _result(successes: list[ToolSuccess], errors: list[ToolError]) -> ToolResult:
    status = "partial" if successes and errors else "failed" if errors else "completed"
    return ToolResult(status=status, successes=successes, errors=errors)


class EntityStore:
    def __init__(self, root: Path | str = ".memory") -> None:
        self.root = Path(root).absolute()

    def _check_root(self, *, create: bool = False) -> None:
        if self.root.is_symlink():
            raise EntityDataError("实体目录不得是符号链接")
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        if self.root.exists() and not self.root.is_dir():
            raise EntityDataError("实体目录路径不是目录")

    def _path(self, entity_id: str) -> Path:
        self._check_root()
        path = self.root / f"{validate_entity_id(entity_id)}.json"
        if path.is_symlink():
            raise EntityDataError(f"拒绝符号链接实体文件：{entity_id}.json")
        return path

    def _read_entity(self, entity_id: str) -> dict[str, Any]:
        path = self._path(entity_id)
        try:
            with path.open("r", encoding="utf-8") as source:
                data = json.load(source, object_pairs_hook=_unique_json_object)
            entity = Entity.model_validate(data).model_dump()
            if entity["id"] != entity_id:
                raise EntityDataError("文件名与内容 ID 不一致")
            return entity
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeError, OSError) as exc:
            raise EntityDataError(f"实体文件 {path.name} 数据错误：{exc}") from exc

    def _check_references(self, entity: dict[str, Any]) -> None:
        pending = [entity]
        checked = {entity["id"]}
        while pending:
            current = pending.pop()
            for targets in current["relations"].values():
                for entity_id in split_values(targets):
                    if entity_id in checked:
                        continue
                    try:
                        target = self._read_entity(entity_id)
                    except FileNotFoundError as exc:
                        raise EntityDataError(
                            f"关系引用的实体不存在：{entity_id}"
                        ) from exc
                    checked.add(entity_id)
                    pending.append(target)

    def get(self, entity_id: str) -> dict[str, Any]:
        entity = self._read_entity(entity_id)
        self._check_references(entity)
        return entity

    def _write(self, entity: dict[str, Any], *, create: bool) -> None:
        """Publish a complete file atomically, without overwriting on creation."""
        self._check_root(create=True)
        path = self._path(entity["id"])
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".entity-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                json.dump(entity, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            # Recheck the target immediately before publishing. Concurrent writers
            # are outside this single-process store's contract.
            self._path(entity["id"])
            if create:
                os.link(temporary, path)
            else:
                if not path.is_file():
                    raise FileNotFoundError(f"待更新实体不存在：{entity['id']}")
                os.replace(temporary, path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # Cleanup cannot turn an already published file into a
                    # reported failed write or mask the original I/O error.
                    pass

    def initialize(self) -> dict[str, Any]:
        self._check_root(create=True)
        try:
            return self.get("agent")
        except FileNotFoundError:
            entity = {
                "id": "agent",
                "keywords": "",
                "agent_cognition": "我是一个智能体。",
                "relations": {},
            }
            self._write(entity, create=True)
            return entity

    def query_entities(
        self, ids: list[str] | None = None, keywords: list[str] | None = None
    ) -> ToolResult:
        try:
            query = QueryEntitiesInput.model_validate(
                {
                    "ids": [] if ids is None else ids,
                    "keywords": [] if keywords is None else keywords,
                }
            )
        except ValueError as exc:
            return _result([], [ToolError(step="validation", message=str(exc))])
        try:
            self._check_root()
            # A keyword scan must surface corrupt files rather than present them
            # as absent matches, which could otherwise cause duplicate creation.
            candidates = (
                sorted(self.root.glob("*.json"))
                if query.keywords
                else [self._path(entity_id) for entity_id in sorted(query.ids)]
            )
            matches: dict[str, dict[str, Any]] = {}
            for path in candidates:
                try:
                    entity = self.get(path.stem)
                except FileNotFoundError:
                    if query.keywords:
                        raise
                    continue
                terms = [term.casefold() for term in split_values(entity["keywords"])]
                if entity["id"] in query.ids or any(
                    keyword in term for keyword in query.keywords for term in terms
                ):
                    matches[entity["id"]] = entity
            return _result(
                [
                    ToolSuccess(index=index, data=matches[entity_id])
                    for index, entity_id in enumerate(sorted(matches))
                ],
                [],
            )
        except (ValueError, OSError) as exc:
            return _result([], [ToolError(step="query", message=str(exc))])

    def create_entities(self, entities: list[dict[str, Any]]) -> ToolResult:
        try:
            batch = CreateEntitiesInput.model_validate({"entities": entities})
        except ValueError as exc:
            return _result([], [ToolError(step="validation", message=str(exc))])
        successes, errors = [], []
        for index, initial in enumerate(batch.entities):
            entity = {"id": str(uuid4()), **initial.model_dump()}
            try:
                self._write(entity, create=True)
                successes.append(ToolSuccess(index=index, data=entity))
            except (ValueError, OSError) as exc:
                errors.append(ToolError(index=index, step="create", message=str(exc)))
        return _result(successes, errors)

    def update_entities(self, updates: list[dict[str, Any]]) -> ToolResult:
        try:
            batch = UpdateEntitiesInput.model_validate({"updates": updates})
        except ValueError as exc:
            return _result([], [ToolError(step="validation", message=str(exc))])
        prepared: list[tuple[dict[str, Any], bool]] = []
        for index, update in enumerate(batch.updates):
            try:
                original = self.get(update.id)
                entity = {
                    **original,
                    **update.attributes,
                    "relations": dict(original["relations"]),
                }
                for attribute in update.remove_attributes:
                    entity.pop(attribute, None)
                # Validate targets for both addition and removal, even if removal
                # is idempotent. Invalid references never reach the write phase.
                for relation_map in (update.relations_add, update.relations_remove):
                    for targets in relation_map.values():
                        for entity_id in split_values(targets):
                            self.get(entity_id)
                for name, targets in update.relations_remove.items():
                    removed = set(split_values(targets))
                    kept = [
                        value
                        for value in split_values(entity["relations"].get(name, ""))
                        if value not in removed
                    ]
                    if kept:
                        entity["relations"][name] = ",".join(kept)
                    else:
                        entity["relations"].pop(name, None)
                for name, targets in update.relations_add.items():
                    entity["relations"][name] = ",".join(
                        dict.fromkeys(
                            split_values(entity["relations"].get(name, ""))
                            + split_values(targets)
                        )
                    )
                entity = Entity.model_validate(entity).model_dump()
                self._check_references(entity)
                prepared.append((entity, entity != original))
            except (ValueError, OSError) as exc:
                return _result(
                    [], [ToolError(index=index, step="validation", message=str(exc))]
                )
        successes, errors = [], []
        for index, (entity, changed) in enumerate(prepared):
            try:
                if changed:
                    self._write(entity, create=False)
                successes.append(ToolSuccess(index=index, data=entity))
            except (ValueError, OSError) as exc:
                errors.append(ToolError(index=index, step="update", message=str(exc)))
        return _result(successes, errors)
