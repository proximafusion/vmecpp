# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import types
import typing
from collections.abc import Callable, Mapping
from typing import Any, Literal, Union

import jaxtyping as jt
import numpy as np
import pydantic


class BaseModelWithNumpy(pydantic.BaseModel):
    """A minimal layer on top of pydantic to help with serialization and de-
    serialization of classes with numpy arrays, annotated with jaxtyping.

    Does checks on the shape and type of arrays, may do casting if needed.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        # Serialize NaN and infinite floats as strings in JSON output.
        # Due to a bug in Pydantic, this setting is ignored by model_dump(mode="json"),
        # so we override it below to also convert values to string there.
        ser_json_inf_nan="strings",
    )

    @pydantic.field_serializer("*", mode="wrap", when_used="json")
    def _serialize_field(
        self,
        value: typing.Any,
        default_handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.FieldSerializationInfo,
        # Note: Do NOT annotate return type with "-> Any" here, since that removes types
        # from the JSON schema of all fields (in serialization mode).
    ):
        value = serialize_special_field(type(self), info.field_name, value)
        return default_handler(value)

    @pydantic.field_validator("*", mode="wrap")
    @classmethod
    def _validate_field(
        cls,
        value: typing.Any,
        default_handler: pydantic.ValidatorFunctionWrapHandler,
        info: pydantic.ValidationInfo,
    ) -> typing.Any:
        assert info.field_name is not None
        # We consciously *ignore* info.mode_is_json() here to allow validating from
        # JSON-like dicts without having to go through strings. This is consistent with
        # Pydantic's default behavior: while serializers retain Python objects in
        # model_dump, but convert them to JSON values in model_dump_json, validators
        # are lenient and accept JSON values in both modes.
        value = deserialize_special_field(cls, info.field_name, value)
        return default_handler(value)

    # This override is necessary to make also `model_dump(mode="json")` respect the
    # ser_json_inf_nan="strings" setting in model_config. Without this fix, Pydantic
    # would keep returning NaN/Inf as Python floats from this function, which leads to
    # invalid JSON, e.g. when the output is used with `json.dumps`.
    # This bug will probably only be fixed in Pydantic V3, since they consider it a
    # breaking change.
    # See: https://github.com/pydantic/pydantic/issues/10037#issuecomment-2314751795
    # Hide the override from the type system to "pass through" docstring and parameter
    # types and defaults.
    if not typing.TYPE_CHECKING:

        def model_dump(self, *, mode: str = "python", **kwargs: Any) -> dict[str, Any]:
            output_dict = super().model_dump(mode=mode, **kwargs)
            if mode == "json":
                sanitize_floats_in_container(output_dict)
            return output_dict


"""This module handles special cases for serialization of BaseModelWithNumpy types.

Any data type that pydantic cannot natively handle can be supported by
adding a case in this module.

Only add special cases here if the type is not under your control (it comes from
a third-party library) AND if it is of broad interest across the codebase.

Before adding a special case to this module, consider one of these two alternatives:

1) Wrap the type definition and add Pydantic serializers and validators directly:
   `Annotated[<type>, pydantic.PlainSerializer(...), pydantic.BeforeValidator(...)]`
   Example can be found here: vmecpp/src/vmecpp/__init__.py

2) Convert/wrap the unsupported class in a BaseModelWithNumpy class and provide a
   @pydantic.field_serializer and @pydantic.field_validator method on that class.
   See: https://docs.pydantic.dev/latest/concepts/serialization/#custom-serializers

The advantage of adding a special case to this module is solely so types can be
declared in BaseModelWithNumpy classes "as-is" without wrapping them with Annotated types.
For example, you can declare `value: jt.Float[np.ndarray, ...]` instead of
`value: NumpyArrayWrapper[...]`.

----

The mechanism for serialization is to convert "special" fields to JSON-serializable
values and then let Pydantic take care of the rest.
In practice:
- If the field is "special", (e.g. `np.ndarray`) it is converted to either a primitive
  list.
- If the field is a generic/composed type such as a list, an optional or a union,
  recurse and do the same for the inner types.
- Otherwise just return the field as is.

The mechanism for deserialization is the inverse: Pydantic provides the serialized field
and we convert it back to the original type (e.g. `np.ndarray`) while returning
the rest unchanged to Pydantic's deserialization logic.

----

JAX array handling works as follows:
 - During serialization, JAX arrays are converted to numpy arrays and serialized with
   the same rules.
 - During validation, "jt.Array" annotations are NOT treated specially, thus only numpy
   arrays can be produced by the deserialization process. To put JAX arrays into
   BaseModelWithNumpy types that should be serializable, they must be annotated with
   np.ndarray | jt.Array!
"""
NpOrAbstractArray = np.ndarray | jt.AbstractArray

AnyJsonValue = dict | list | str | int | float | bool | None
"""Type alias for any value that can be part of a JSON object."""

_NUMPY_ALLOWED_NONBINARY_DTYPES = (np.float64, np.int64, np.bool_)
"""Allowed numpy dtypes for serialization without the `Binary` annotation."""


def serialize_special_field(
    cls: type[pydantic.BaseModel],
    field_name: str,
    value: Any,
) -> Any:
    """Preprocesses a field value, serializing special types while leaving others
    unchanged."""
    field_info = cls.model_fields.get(field_name, None)
    if field_info is not None:
        assert field_info.annotation is not None
        field_type = field_info.annotation
        field_metadata = field_info.metadata
    else:
        assert (
            field_name in cls.model_computed_fields
        ), f"Field {field_name} must be either a field or a computed field."
        computed_field_info = cls.model_computed_fields[field_name]
        field_type = computed_field_info.return_type
        field_metadata = []  # Computed fields cannot use Annotated[...] for now.

    try:
        return _traverse_field_contents(
            field_type, value, _serialize_value, field_metadata
        )
    except ValueError as e:
        msg = f'At field "{field_name}" of type {type(value).__name__}: {e}'
        raise ValueError(msg)  # noqa: B904


def deserialize_special_field(
    cls: type[pydantic.BaseModel],
    field_name: str,
    value: Any,
) -> Any:
    """Preprocesses a serialized field value, deserializing special types while leaving
    others unchanged."""
    # Note: Computed fields do not call field validators, so we do not need to consider
    # them for deserialization.
    field_info = cls.model_fields[field_name]
    assert field_info.annotation is not None
    return _traverse_field_contents(
        field_info.annotation, value, _deserialize_value, field_info.metadata
    )


def _is_arraylike(x):
    return hasattr(x, "__array__") or hasattr(x, "__array_interface__")


def _serialize_value(declared_type: type, value: Any, field_metadata: list[Any]) -> Any:  # noqa: ARG001
    """Tests for the presence of a special type and converts it to a serializable value.

    Args:
        declared_type: The innermost declared type of the field. Unused, since for
            serialization, the runtime type of the value is enough.
        value: The scalar value to serialize, that potentially needs special handling.

    Returns:
        Any value that can be serialized by Pydantic, for example a dict representation
        of the object, or a raw string.
        Returns the unmodified value if no special handling applies.
    """
    if issubclass(declared_type, NpOrAbstractArray) and _is_arraylike(value):
        # Convert JAX arrays to numpy arrays for the purpose of serialization.
        np_array = np.asarray(value)
        if np_array.dtype in _NUMPY_ALLOWED_NONBINARY_DTYPES:
            return np_array.tolist()
        msg = (
            f"Cannot serialize numpy array with dtype {np_array.dtype} to JSON. "
            f"Please convert it to an allowed dtype: {_NUMPY_ALLOWED_NONBINARY_DTYPES}."
        )
        raise ValueError(msg)
    return value


def _deserialize_value(
    declared_type: type, value: Any, field_metadata: list[Any]
) -> Any:
    """Tests for the presence of a special type and converts its serialized value back
    to the original representation.

    This function must gracefully handle values that are NOT valid for the declared
    type, since deserialization is attempted for Union types where one of the options is
    a special type, but the concrete value is not.

    This function must also handle values while validating in "Python" mode using
    "model_validate", since that function must support both raw JSON values and higher-
    level Python values. Since these values already have the correct data type, they
    should be returned unchanged.

    Args:
        declared_type: The innermost declared type of the field, after unwrapping
            containers and unions from its declaration.
        value: The serialized value, either in its JSON form or as a Python object.
        field_metadata: The annotation metadata of the field.

    Returns:
        The deserialized value, or the unmodified value if it is not a special type.
    """
    del field_metadata  # unused

    if issubclass(declared_type, NpOrAbstractArray) and isinstance(value, list):
        fixed_list = _reconstruct_floats_for_numpy(value)
        np_data = np.array(fixed_list)
        # Only accept allowed dtypes, to avoid misinterpreting non-numpy lists.
        if np_data.dtype in _NUMPY_ALLOWED_NONBINARY_DTYPES:
            return np_data
    return value


def _traverse_field_contents(
    field_type: type,
    value: Any,
    value_converter: Callable[[type, Any, list[Any]], Any],
    field_metadata: list[Any],
) -> Any:
    """Traverses the insides of a field's declared type, recursively resolving nested
    and collection types and returning the processed result.

    This is called for both serialization and deserialization.
    """
    # Fields declared as Any are not processed, since their deserialization is under-
    # specified. If we would allow applying transformations at serialization time, there
    # would be no way to reverse it during deserialization without a "target" type hint.
    if field_type is Any:
        return value

    outer_type = typing.get_origin(field_type)

    # Performance note: The following code recurses into containers (lists, dicts, ...)
    # looking for special values, independently of their declared value type.
    # If this becomes a performance bottleneck, consider adding a check here to skip
    # containers that are known to not contain special values (e.g. list[float]).
    # Simple type.
    if outer_type is None:
        return value_converter(field_type, value, field_metadata)

    # Annotated types are a special case, since they can be used to add metadata
    # to the field. We need to check if the first argument is a special type.
    # The rest of the arguments are passed to the value converter.
    # NOTE: This branch only handles "Annotated" if it appears nested inside other
    # declarations. If it is the outermost type, Pydantic strips it automatically and
    # its annotations will already be placed inside "field_metadata".
    if outer_type is typing.Annotated:
        inner_types = typing.get_args(field_type)
        inner_type = inner_types[0]
        # While processing this branch, add the annotations to the metadata list.
        new_field_metadata = field_metadata + list(inner_types[1:])
        return _traverse_field_contents(
            inner_type, value, value_converter, new_field_metadata
        )

    # Literal[X, Y, ...] cannot be used with issubclass and would fail the other checks,
    # so we need to check it separately.
    if outer_type is Literal:
        return value

    # Defensive check: Some built-in types are not proper classes and for those,
    # "issubclass" will raise an exception. It's hard to decide in general what should
    # happen to those and if more cases even exist, so let's be defensive and admit that
    # we don't know how to handle them ...
    if not isinstance(outer_type, type) and outer_type is not Union:
        msg = f"BaseModelWithNumpy serialization does not know how to handle type {outer_type}! "
        raise NotImplementedError(msg)

    # - Union[T, TOther].
    # - Optional[T] becomes Union[T, None].
    # - Surprisingly, "int|str" becomes types.UnionType[int, str] instead of Union,
    #   see: https://github.com/python/cpython/issues/105499
    if outer_type is Union or outer_type is types.UnionType:
        # Unions are complex to deserialize, since it must be deduced which of the
        # declared types to deserialize to. For example, it is unclear how to handle
        # a field `Union[np.ndarray, Blob]`, since both types share the same JSON
        # structure.
        # Pydantic has complex heuristics for this, but we haven't reimplemented them
        # fully. See: https://docs.pydantic.dev/2.7/concepts/unions/
        # Our approach is similar to Pydantic's "left-to-right" mode, but doesn't handle
        # validation errors perfectly when multiple similar-looking Union types COULD
        # match the given JSON data. If this becomes a problem and validation fails when
        # it shouldn't, this logic should be revisited in the future.
        # ----
        # In serialization mode, the task is easier: The first time value_converter
        # is called, it can directly decide whether or not it needs to handle a special
        # case. This function still checks all options just to keep the code simpler.

        result = value
        inner_types = typing.get_args(field_type)
        for value_type in inner_types:
            result = _traverse_field_contents(
                value_type, value, value_converter, field_metadata
            )
            # Stop at the first conversion that changed something, similar to Pydantic's
            # "left-to-right" mode.
            if result is not value:
                return result
        return result
    outer_type = typing.cast(type, outer_type)
    # Dicts and similar.
    if issubclass(outer_type, Mapping) and isinstance(value, Mapping):
        value_type = typing.get_args(field_type)[1]
        return outer_type(
            {
                key: _traverse_field_contents(
                    value_type, item, value_converter, field_metadata
                )
                for key, item in value.items()
            }  # type: ignore
        )

    # Tuples: They can be declared as tuple[T1, T2] or tuple[T, ...].
    # During deserialization, `value` will however be a list!
    if issubclass(outer_type, tuple) and isinstance(value, tuple | list):
        value_types = typing.get_args(field_type)
        if len(value_types) == 2 and value_types[1] is Ellipsis:
            value_types = [value_types[0]] * len(value)
        return outer_type(
            _traverse_field_contents(item_type, item, value_converter, field_metadata)
            for item_type, item in zip(value_types, value, strict=True)
        )

    # Lists and similar.
    # Cannot use "Sequence" since that would also capture "str" and maybe others.
    if issubclass(outer_type, list | set | frozenset) and isinstance(
        value, list | set | frozenset
    ):
        value_type = typing.get_args(field_type)[0]
        return outer_type(
            _traverse_field_contents(value_type, item, value_converter, field_metadata)
            for item in value
        )

    # Other generic type. We don't support these, just pass them through.
    return value


def sanitize_floats_in_container(value: list | dict) -> None:
    """Traverses a JSON container and converts unsupported float values to strings.

    This function operates in-place to avoid creating a copy of the whole structure.
    """
    # Iterate through either (i, item) or (key, item) pairs.
    tuple_iterator = enumerate(value) if isinstance(value, list) else value.items()
    for key, item in tuple_iterator:
        # Recurse into other containers.
        if isinstance(item, list | dict):
            sanitize_floats_in_container(item)
        # In-place conversion for floats.
        elif isinstance(item, float):
            value[key] = _sanitize_float(item)


def _sanitize_float(value: float) -> float | str:
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "-Infinity" if value < 0 else "Infinity"
    return value


def _reconstruct_floats_for_numpy(value: list | float | str) -> list | float | str:
    """Reconstructs floats in a (potentially nested) list so numpy can handle them.

    While pydantic can coerce "NaN" and "Infinity" back to floats, our custom numpy
    deserialization from a Python list cannot reuse this logic, so we need to check for
    stringified values ourselves.

    Only lists are supported as containers, since that's what we need for `np.array`.
    """
    if isinstance(value, list):
        return [_reconstruct_floats_for_numpy(item) for item in value]
    return _reconstruct_float(value)


def _reconstruct_float(value: float | str) -> float | str:
    # We handle some other spellings as well, to stay consistent with pydantic's
    # built-in validation logic.
    if value == "NaN" or value == "nan" or value is None:  # noqa: PLR1714
        return np.nan
    if value == "Infinity" or value == "inf":  # noqa: PLR1714
        return np.inf
    if value == "-Infinity" or value == "-inf":  # noqa: PLR1714
        return -np.inf
    return value
