"""``Annotated[...]`` → click options: the ``with_params`` dataclass-click mechanism.

**Leaf-module invariant: this module MUST NOT import anything from
``ao_shaping``** — not even indirectly. It is the leaf that both ``runners/``
and ``tools/`` depend on, and its only third-party dependency is ``click``.

Why the constraint is load-bearing (do not relax it):

* ``runners/slm/zernike_matrix_runner.py`` imports ``ao_shaping.tools.slm.*`` at
  module level, so ``tools/slm`` must never import ``runners/``. Because this
  module sits below both, a single ``from ao_shaping...`` here would open the
  cycle ``utils.cli.params → ao_shaping.<pkg> → ... → runners.runner_common →
  utils.cli.params``.
* The mechanism is *pure metadata plumbing*: delayed click options, annotation
  introspection, and option-name/type/default patching. None of it needs a
  driver, an optimizer, a dataclass from the repo, or ``loguru`` — which is
  exactly why it can live this low.

The convention
--------------
Each parameter field is declared ``name: Annotated[T, option(...)] = default``.
The dataclass field default is the single source of truth for the CLI default —
``with_params`` injects it into the click option (an explicit ``default=``
inside ``option(...)`` raises ``TypeError``). ``option`` is a delayed
``click.option``: inside ``Annotated`` it returns a ``_DelayedCall`` that
``with_params`` applies to the command in *reversed* declaration order, so the
CLI help lists the options top-to-bottom in the same order the class reads.

Option names come from the declaration, flags first (e.g.
``option("-c", "--center")``) — never repeat the field name as the first
positional. The click type is inferred from the field annotation
(``str``/``int``/``float``/``bool``/``Path``; ``Optional[x]`` / ``x | None``
strips to ``x``) unless ``type=``, ``callback=``, ``is_flag`` or ``multiple``
is given explicitly. A union of two concrete types (e.g.
``str | tuple[int, int]``) MUST pass an explicit ``type=``.

The wrapped command receives one keyword argument per decorator, named by
``kw_name``, holding a fully-populated instance of the parameter class.

Four fail-fast sites guard the convention — all raise at *import* time when a
consumer declaration drifts, so a mistake cannot reach a hardware run:

* :func:`_strip_optional` — ``TypeError`` unless a union has exactly one
  non-``None`` member.
* :func:`_patch_click_types` — ``TypeError`` for an unmappable bare type.
* :func:`_patch_defaults` — ``TypeError`` if ``default=`` is passed inside
  ``option(...)``.
* :func:`_collect_click_annotations` — ``TypeError`` on a duplicate click
  parameter name within one class tree (a nested :class:`ClickGroup` shadowing
  an existing option name).

Annotations are resolved through :func:`typing.get_type_hints` with
``include_extras=True``, so the dataclasses that carry these fields may use
either eager or PEP 563 (``from __future__ import annotations``) evaluation;
the ``Annotated`` metadata objects are only materialised at decoration time.

Public API re-exported by :mod:`ao_shaping.utils.cli`: :class:`ClickGroup`,
:data:`option`, :func:`with_params`. The private ``_``-prefixed helpers are
internal; tests import ``_collect_click_annotations`` from this module.
"""

from __future__ import annotations

import functools
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import click

# ---------------------------------------------------------------------------
# dataclass-click machinery (B2 copy of the dataclass-click convention:
# Annotated[...] metadata + with_params collector, object delivery)
# ---------------------------------------------------------------------------


class _DelayedCall:
    """A ``click.option`` declaration captured but not yet applied."""

    __slots__ = ("callable", "args", "kwargs")

    def __init__(self, callable: object, args: tuple, kwargs: dict[str, Any]) -> None:
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged = dict(self.kwargs)
        merged.update(kwargs)
        return self.callable(*self.args, *args, **merged)


class _DelayedFunction:
    """Wrap ``click.option`` so it can be invoked inside ``Annotated[...]``."""

    def __init__(self, fn: object) -> None:
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> _DelayedCall:
        return _DelayedCall(self.fn, args, kwargs)


option = _DelayedFunction(click.option)


_TYPE_INFERENCE: dict[type[Any], click.ParamType] = {
    str: click.STRING,
    int: click.INT,
    float: click.FLOAT,
    bool: click.BOOL,
    Path: click.Path(path_type=Path),
}


def _strip_optional(tp: Any) -> Any:
    if get_origin(tp) in (Union, UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(
                f"Cannot infer click type from union {tp!r} — pass an explicit type=."
            )
        return args[0]
    return tp


def _patch_names(decl: tuple, name: str) -> tuple:
    # Prepend the field name so click maps user input back onto the field.
    # Declarations must therefore be flags-first (no duplicated first positional).
    return (name, *decl)


def _patch_click_types(name: str, field_type: Any, kwargs: dict[str, Any]) -> None:
    if (
        "type" in kwargs
        or "callback" in kwargs
        or kwargs.get("is_flag")
        or kwargs.get("multiple")
    ):
        return
    stripped = _strip_optional(field_type)
    try:
        kwargs["type"] = _TYPE_INFERENCE[stripped]
    except KeyError:
        raise TypeError(
            f"Field {name!r}: cannot infer click type from {field_type!r} — pass an explicit type=."
        ) from None


def _patch_defaults(name: str, field: object, kwargs: dict[str, Any]) -> None:
    if "default" in kwargs:
        raise TypeError(
            f"Field {name!r}: default must live on the dataclass field, not in option()."
        )
    value = field.default
    if value is not MISSING:
        kwargs["default"] = value


def _copy_delayed_call(d: _DelayedCall) -> _DelayedCall:
    return _DelayedCall(
        d.callable,
        tuple(d.args),
        {k: (list(v) if isinstance(v, list) else v) for k, v in d.kwargs.items()},
    )


@dataclass(frozen=True)
class ClickGroup:
    """Opt-in marker for a field whose value is a nested parameter dataclass.

    Annotate the field as ``Annotated[SomeParams, ClickGroup()]``: every
    ``option(...)`` declared inside ``SomeParams`` is hoisted onto the parent
    command as a flat option, then re-assembled into a ``SomeParams`` instance
    before the wrapped callback runs. Unmarked fields keep the previous
    strictly-flat behaviour, so existing runners are unaffected.

    A frozen dataclass (rather than a ``dict``) is used as the marker so the
    ``Annotated[...]`` alias stays hashable and its identity stable.
    """


def _is_click_group(hint: Any) -> bool:
    return get_origin(hint) is Annotated and any(
        isinstance(m, ClickGroup) for m in get_args(hint)[1:]
    )


# (dataclass field supplying the default, click parameter name, type hint, option())
_ClickOption = tuple[Any, str, Any, _DelayedCall]
# (parent field name, nested dataclass, ordered child parameter names)
_ClickGroupSpec = tuple[str, Any, list[str]]


def _collect_click_annotations(
    cls: type,
) -> tuple[list[_ClickOption], list[_ClickGroupSpec]]:
    """Return ``(options, groups)`` for ``cls`` in command-help order.

    A :class:`ClickGroup` field contributes its nested options at the position
    where the group itself is declared, so the flat help order still follows
    the parent's declaration order.
    """
    hints = get_type_hints(cls, include_extras=True)
    options: list[_ClickOption] = []
    groups: list[_ClickGroupSpec] = []
    seen: dict[str, str] = {}

    def _add(fld: Any, hint: Any, owner: str) -> None:
        if fld.name in seen:
            raise TypeError(
                f"{cls.__name__}: click parameter {fld.name!r} is declared twice "
                f"({seen[fld.name]} and {owner}); a nested ClickGroup must not "
                f"shadow an existing option name."
            )
        seen[fld.name] = owner
        delayed = next(m for m in get_args(hint)[1:] if isinstance(m, _DelayedCall))
        options.append((fld, fld.name, get_args(hint)[0], delayed))

    for f in fields(cls):
        hint = hints.get(f.name)
        if hint is None or get_origin(hint) is not Annotated:
            continue
        if _is_click_group(hint):
            child_cls = get_args(hint)[0]
            child_hints = get_type_hints(child_cls, include_extras=True)
            child_params: list[str] = []
            for cf in fields(child_cls):
                chint = child_hints.get(cf.name)
                if chint is None or get_origin(chint) is not Annotated:
                    continue
                if not any(isinstance(m, _DelayedCall) for m in get_args(chint)[1:]):
                    continue
                _add(cf, chint, f"{cls.__name__}.{f.name}")
                child_params.append(cf.name)
            groups.append((f.name, child_cls, child_params))
            continue
        if not any(isinstance(m, _DelayedCall) for m in get_args(hint)[1:]):
            continue
        _add(f, hint, cls.__name__)
    return options, groups


def with_params(arg_class: type, *, kw_name: str) -> Any:
    """Attach the click options of ``arg_class`` to a click command.

    Every ``Annotated[..., option(...)]`` field becomes a click option (help
    text order == dataclass declaration order). Fields marked with
    :class:`ClickGroup` additionally contribute their nested options, which are
    re-assembled into the nested instance before delivery. The wrapped command
    receives ``kw_name=arg_class(**provided_fields)`` — one keyword argument per
    decorator, containing a fully-populated parameter instance.
    """
    options, groups = _collect_click_annotations(arg_class)

    def decorator(fn: Any) -> Any:
        # Apply in REVERSED declaration order so click's cumulative
        # __click_params__ yields help in the same order the class reads.
        for fld, param, field_type, delayed in reversed(options):
            dc = _copy_delayed_call(delayed)
            dc.args = _patch_names(delayed.args, param)
            _patch_click_types(param, field_type, dc.kwargs)
            _patch_defaults(param, fld, dc.kwargs)
            fn = dc.callable(*dc.args, **dc.kwargs)(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            obj_kwargs: dict[str, Any] = {}
            for _fld, param, _field_type, _delayed in options:
                # Membership (not truthiness) so 0 / 0.0 / "" survive.
                if param in kwargs:
                    obj_kwargs[param] = kwargs.pop(param)
            for group_field, group_cls, child_params in groups:
                child_kwargs = {
                    p: obj_kwargs.pop(p) for p in child_params if p in obj_kwargs
                }
                obj_kwargs[group_field] = group_cls(**child_kwargs)
            kwargs[kw_name] = arg_class(**obj_kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
