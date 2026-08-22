from __future__ import annotations

from datetime import datetime
from typing import Any, Callable
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, MetaData, Uuid, event, func, inspect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.types import TypeDecorator


naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=naming_convention)


class PlatformJSON(TypeDecorator[Any]):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


JsonType = PlatformJSON


def _coerce_json_value(value: Any) -> Any:
    if isinstance(value, (RecursiveMutableDict, RecursiveMutableList)):
        return value
    if isinstance(value, dict):
        return RecursiveMutableDict(value)
    if isinstance(value, list):
        return RecursiveMutableList(value)
    return value


def _propagate_mutable_change(container: Any) -> None:
    for parent, key in list(container._parents.items()):
        if isinstance(parent, (RecursiveMutableDict, RecursiveMutableList)):
            parent.changed()
        else:
            flag_modified(parent.obj(), key)


class RecursiveMutableDict(MutableDict):
    __hash__ = object.__hash__

    @classmethod
    def coerce(cls, key: str, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value)
        return super().coerce(key, value)

    def __init__(self, value: dict[str, Any] | None = None) -> None:
        dict.__init__(self)
        if value:
            self.update(value)

    def __setitem__(self, key: str, value: Any) -> None:
        value = _coerce_json_value(value)
        dict.__setitem__(self, key, value)
        if isinstance(value, (RecursiveMutableDict, RecursiveMutableList)):
            value._parents[self] = key
        self.changed()

    def update(self, value: Any = (), **kwargs: Any) -> None:
        items = value.items() if hasattr(value, "items") else value
        for key, item in items:
            self[key] = item
        for key, item in kwargs.items():
            self[key] = item

    def changed(self) -> None:
        _propagate_mutable_change(self)


class RecursiveMutableList(MutableList):
    __hash__ = object.__hash__

    @classmethod
    def coerce(cls, key: str, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, list):
            return cls(value)
        return super().coerce(key, value)

    def __init__(self, value: list[Any] | None = None) -> None:
        list.__init__(self)
        if value:
            self.extend(value)

    def _attach(self, index: int, value: Any) -> Any:
        value = _coerce_json_value(value)
        if isinstance(value, (RecursiveMutableDict, RecursiveMutableList)):
            value._parents[self] = index
        return value

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            values = [self._attach(position, item) for position, item in enumerate(value)]
            list.__setitem__(self, index, values)
        else:
            list.__setitem__(self, index, self._attach(index, value))
        self.changed()

    def append(self, value: Any) -> None:
        list.append(self, self._attach(len(self), value))
        self.changed()

    def extend(self, values: Any) -> None:
        for value in values:
            self.append(value)

    def insert(self, index: int, value: Any) -> None:
        list.insert(self, index, self._attach(index, value))
        self.changed()

    def __delitem__(self, index: Any) -> None:
        list.__delitem__(self, index)
        self.changed()

    def changed(self) -> None:
        _propagate_mutable_change(self)


class UUIDPrimaryKeyMixin:
    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


def json_column(default: Any = dict) -> Any:
    mutable_json = (
        RecursiveMutableList.as_mutable(PlatformJSON())
        if default is list
        else RecursiveMutableDict.as_mutable(PlatformJSON())
    )
    return mapped_column(mutable_json, nullable=False, default=default)


class ImmutableFieldError(ValueError):
    """Raised when an immutable generated field is changed after insertion."""


def install_immutable_guard(
    model: type[Base],
    fields: set[str],
    *,
    seal_field: str | None = None,
    seal_check: Callable[[Base], bool] | None = None,
    allow_initial_null_fields: set[str] | None = None,
    allow_initial_empty_fields: set[str] | None = None,
    allow_initial_empty_check: Callable[[Base], bool] | None = None,
) -> None:
    model.__immutable_fields__ = fields
    model.__immutable_seal_field__ = seal_field
    model.__immutable_seal_check__ = seal_check
    model.__immutable_allow_initial_null_fields__ = allow_initial_null_fields or set()
    model.__immutable_allow_initial_empty_fields__ = allow_initial_empty_fields or set()
    model.__immutable_allow_initial_empty_check__ = allow_initial_empty_check


def _seal_state(target: Base) -> bool:
    seal_check = getattr(type(target), "__immutable_seal_check__", None)
    if seal_check is not None:
        return bool(seal_check(target))
    seal_field = getattr(type(target), "__immutable_seal_field__", None)
    return bool(getattr(target, seal_field, False)) if seal_field else True


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, UUID):
        return str(value)
    return value


def _immutable_snapshot(target: Base) -> dict[str, Any]:
    return {
        "sealed": _seal_state(target),
        "fields": {
            field: _canonical(getattr(target, field))
            for field in getattr(type(target), "__immutable_fields__", set())
        },
    }


def _store_immutable_snapshot(target: Base) -> None:
    if getattr(type(target), "__immutable_fields__", None):
        target.__immutable_snapshot__ = _immutable_snapshot(target)


def _check_immutable_snapshot(target: Base) -> None:
    previous = getattr(target, "__immutable_snapshot__", None)
    if previous is None:
        return
    current = _immutable_snapshot(target)
    if previous["sealed"] and not current["sealed"]:
        raise ImmutableFieldError(
            f"{target.__class__.__name__}.definition_sealed can only be enabled once"
        )
    empty_fields = {
        field
        for field, old_value in previous["fields"].items()
        if current["fields"][field] != old_value
        and field in getattr(type(target), "__immutable_allow_initial_empty_fields__", set())
        and old_value == ()
    }
    empty_check = getattr(type(target), "__immutable_allow_initial_empty_check__", None)
    blocked_empty_fields = empty_fields if empty_check is None or not empty_check(target) else set()
    if blocked_empty_fields:
        raise ImmutableFieldError(
            f"{target.__class__.__name__} initial metrics write requires pending status"
        )
    changed = [
        field
        for field, old_value in previous["fields"].items()
        if current["fields"][field] != old_value
        and not (
            field in getattr(type(target), "__immutable_allow_initial_null_fields__", set())
            and old_value is None
            and getattr(target, field) is not None
        )
            and not (
                field in getattr(type(target), "__immutable_allow_initial_empty_fields__", set())
                and old_value == ()
                and current["fields"][field] != ()
                and field not in blocked_empty_fields
            )
    ]
    if previous["sealed"] or current["sealed"]:
        if changed:
            raise ImmutableFieldError(
                f"{target.__class__.__name__} fields are immutable after sealing: "
                f"{', '.join(sorted(changed))}"
            )


@event.listens_for(Session, "loaded_as_persistent")
def _snapshot_loaded_object(session: Session, target: Base) -> None:
    _store_immutable_snapshot(target)


@event.listens_for(Session, "before_flush")
def _guard_immutable_objects(
    session: Session, flush_context: Any, instances: Any
) -> None:
    for target in session.identity_map.values():
        if not inspect(target).deleted:
            _check_immutable_snapshot(target)


@event.listens_for(Session, "before_commit")
def _guard_immutable_objects_before_commit(session: Session) -> None:
    with session.no_autoflush:
        for target in session.identity_map.values():
            if not inspect(target).deleted:
                _check_immutable_snapshot(target)


@event.listens_for(Session, "after_flush_postexec")
def _snapshot_flushed_objects(session: Session, flush_context: Any) -> None:
    for target in session.identity_map.values():
        if not inspect(target).deleted:
            _store_immutable_snapshot(target)
