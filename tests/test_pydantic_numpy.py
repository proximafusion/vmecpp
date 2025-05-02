# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MITimport datetime as dt
import datetime as dt
import json
from collections.abc import Callable
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pydantic
import pytest

from vmecpp._pydantic_numpy import BaseModelWithNumpy

jax.config.update("jax_enable_x64", True)


class ModelPlainArray(BaseModelWithNumpy):
    np_array: np.ndarray


class ModelJaxtypingArray(BaseModelWithNumpy):
    np_array: jt.Float[np.ndarray, " n_elements"] | jt.Int[np.ndarray, " n_elements"]


class ModelOptionalArray(BaseModelWithNumpy):
    np_array: np.ndarray | None


class ModelUnion1Array(BaseModelWithNumpy):
    np_array: str | np.ndarray | float


class ModelUnion2Array(BaseModelWithNumpy):
    np_array: str | np.ndarray | float


@pytest.mark.parametrize(
    "ModelClass",
    [
        ModelPlainArray,
        ModelJaxtypingArray,
        ModelOptionalArray,
        ModelUnion1Array,
        ModelUnion2Array,
    ],
)
def test_serialize_numpy_plain(ModelClass: type):
    np_data = np.array([1.0, 2.0, 3.0, 4.2])
    model: BaseModelWithNumpy = ModelClass(np_array=np_data)

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert isinstance(json_obj, dict)
    assert "np_array" in json_obj
    assert json_obj["np_array"] == [1.0, 2.0, 3.0, 4.2]

    deserialized = ModelClass.model_validate_json(serialized)  # type: ignore

    assert isinstance(deserialized.np_array, np.ndarray)
    assert np.all(deserialized.np_array == np_data)


def test_serialize_optional_none():
    class Model(BaseModelWithNumpy):
        np_array: np.ndarray | None

    model = Model(np_array=None)

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert json_obj["np_array"] is None

    deserialized = Model.model_validate_json(serialized)

    assert deserialized.np_array is None


def test_serialize_union_value_before_array():
    class ModelUnionBefore(BaseModelWithNumpy):
        union_str: str | np.ndarray
        union_float: float | np.ndarray
        union_list: list[str] | np.ndarray

    model = ModelUnionBefore(
        union_str="foobar", union_float=42.42, union_list=["foo", "bar"]
    )

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert json_obj["union_str"] == "foobar"
    assert json_obj["union_float"] == 42.42
    assert json_obj["union_list"] == ["foo", "bar"]

    deserialized = ModelUnionBefore.model_validate_json(serialized)

    assert deserialized.union_str == "foobar"
    assert deserialized.union_float == 42.42
    assert deserialized.union_list == ["foo", "bar"]


def test_serialize_union_value_after_array():
    class ModelUnionAfter(BaseModelWithNumpy):
        union_str: np.ndarray | str
        union_float: np.ndarray | float
        union_list: np.ndarray | list[str]

    model = ModelUnionAfter(
        union_str="foobar", union_float=42.42, union_list=["foo", "bar"]
    )

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert json_obj["union_str"] == "foobar"
    assert json_obj["union_float"] == 42.42
    assert json_obj["union_list"] == ["foo", "bar"]

    deserialized = ModelUnionAfter.model_validate_json(serialized)

    assert deserialized.union_str == "foobar"
    assert deserialized.union_float == 42.42
    assert deserialized.union_list == ["foo", "bar"]


def test_serialize_numpy_lists():
    class Model(BaseModelWithNumpy):
        np_list: list[jt.Float[np.ndarray, " n_elements"]]
        np_simple_tuple: tuple[np.ndarray, ...]
        np_hard_tuple: tuple[int, np.ndarray, str]

    np_data = np.array([1.0, 2.0, 3.0, 4.2])
    model = Model(
        np_list=[np_data, np_data, np_data],
        np_simple_tuple=(np_data, np_data, np_data),
        np_hard_tuple=(42, np_data, "foobar"),
    )

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert isinstance(json_obj["np_list"], list)
    assert isinstance(json_obj["np_simple_tuple"], list)
    assert isinstance(json_obj["np_hard_tuple"], list)
    assert json_obj["np_list"][0] == [1.0, 2.0, 3.0, 4.2]
    assert json_obj["np_simple_tuple"][0] == [1.0, 2.0, 3.0, 4.2]
    assert json_obj["np_hard_tuple"][0] == 42
    assert json_obj["np_hard_tuple"][1] == [1.0, 2.0, 3.0, 4.2]
    assert json_obj["np_hard_tuple"][2] == "foobar"

    deserialized = Model.model_validate_json(serialized)

    assert isinstance(deserialized.np_list, list)
    assert len(deserialized.np_list) == 3
    assert np.all(deserialized.np_list[0] == np_data)
    assert len(deserialized.np_simple_tuple) == 3
    assert np.all(deserialized.np_simple_tuple[0] == np_data)
    hard_int, hard_np, hard_str = deserialized.np_hard_tuple
    assert hard_int == 42
    assert np.all(hard_np == np_data)
    assert hard_str == "foobar"


def test_serialize_numpy_dict():
    class Model(BaseModelWithNumpy):
        # let's add optional just to make it harder
        np_dict: dict[str, np.ndarray] | None

    np_data = np.array([1.0, 2.0, 3.0, 4.2])
    model = Model(np_dict={"foo": np_data, "bar": np_data})

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert isinstance(json_obj["np_dict"], dict)
    assert "foo" in json_obj["np_dict"]
    assert "bar" in json_obj["np_dict"]
    assert json_obj["np_dict"]["foo"] == [1.0, 2.0, 3.0, 4.2]
    assert json_obj["np_dict"]["bar"] == [1.0, 2.0, 3.0, 4.2]

    deserialized = Model.model_validate_json(serialized)

    assert isinstance(deserialized.np_dict, dict)
    assert "foo" in deserialized.np_dict
    assert "bar" in deserialized.np_dict
    assert np.all(deserialized.np_dict["foo"] == np_data)
    assert np.all(deserialized.np_dict["bar"] == np_data)


def test_numpy_supports_nested_annotated():
    class Model(BaseModelWithNumpy):
        # If Annotated is not at the "outermost" level, Pydantic does not extract it.
        # We have special logic for this, which this test case covers.
        array_in_union: (
            str | Annotated[np.ndarray, "annotation", "other_annotation"] | None
        )
        dict_of_arrays: dict[
            str,
            Annotated[jt.Int[np.ndarray, "..."], "annotation"],
        ]

    model = Model(
        array_in_union=np.array([5566, 6655], dtype=np.int64),
        dict_of_arrays={"foo": np.array([1, 2, 3], dtype=np.int64)},
    )
    serialized = model.model_dump_json()
    deserialized = Model.model_validate_json(serialized)
    assert isinstance(deserialized.array_in_union, np.ndarray)
    assert deserialized.array_in_union.dtype == np.int64
    assert np.all(deserialized.array_in_union == [5566, 6655])
    assert deserialized.dict_of_arrays["foo"].dtype == np.int64
    assert np.all(deserialized.dict_of_arrays["foo"] == [1, 2, 3])


def test_serialize_deeply_nested():
    class Model(BaseModelWithNumpy):
        np_deep: dict[str, list[tuple[np.ndarray | dt.datetime, str]]] | None

    other_data = dt.datetime(2024, 1, 1, 12, 34, 56)
    np_data = np.array([1.0, 2.0, 3.0, 4.2])
    model = Model(np_deep={"foo": [(other_data, "other"), (np_data, "np")]})

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert isinstance(json_obj["np_deep"], dict)
    assert "foo" in json_obj["np_deep"]
    assert json_obj["np_deep"]["foo"][0] == ["2024-01-01T12:34:56", "other"]
    assert json_obj["np_deep"]["foo"][1] == [[1.0, 2.0, 3.0, 4.2], "np"]

    deserialized = Model.model_validate_json(serialized)

    assert isinstance(deserialized.np_deep, dict)
    assert "foo" in deserialized.np_deep
    foo_list = deserialized.np_deep["foo"]
    assert len(foo_list) == 2
    assert isinstance(foo_list[0], tuple)
    assert isinstance(foo_list[1], tuple)
    first_data, first_str = foo_list[0]
    second_data, second_str = foo_list[1]
    assert first_data == other_data
    assert first_str == "other"
    assert isinstance(second_data, np.ndarray)
    assert np.all(second_data == np_data)
    assert second_str == "np"


def test_serialize_computed_field():
    class Model(BaseModelWithNumpy):
        should_compute_nan: bool

        @pydantic.computed_field
        @property
        def computed_float(self) -> float:
            if self.should_compute_nan:
                return np.nan
            return 42.42

        # In practice, this is kinda weird and should likely be a full field rather than
        # a property, but it is a valid use case.
        @pydantic.computed_field
        @property
        def computed_array(self) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

    model1 = Model(should_compute_nan=False)
    model2 = Model(should_compute_nan=True)

    serialized1 = model1.model_dump_json()
    serialized2 = model2.model_dump_json()

    json_obj1 = json.loads(serialized1)
    json_obj2 = json.loads(serialized2)
    assert json_obj1["should_compute_nan"] is False
    assert json_obj1["computed_float"] == 42.42
    assert json_obj1["computed_array"] == [1.0, 2.0, 3.0]
    assert json_obj2["should_compute_nan"] is True
    assert json_obj2["computed_float"] == "NaN"
    assert json_obj2["computed_array"] == [1.0, 2.0, 3.0]


# Test serialization also in "python" mode.
def test_model_dump_preserves_arrays():
    class Model(BaseModelWithNumpy):
        np_array: np.ndarray

    np_data = np.array([1, 2, 3, 4.2])
    model = Model(np_array=np_data)

    serialized = model.model_dump()

    assert serialized["np_array"] is np_data


def test_model_validate_accepts_arrays():
    class Model(BaseModelWithNumpy):
        np_array: np.ndarray

    np_data = np.array([1, 2, 3, 4.2])

    model = Model.model_validate({"np_array": np_data})

    assert model.np_array is np_data


def test_model_validate_accepts_lists():
    class Model(BaseModelWithNumpy):
        np_array: np.ndarray

    list_data = [1, 2, 3, 4.2]

    model = Model.model_validate({"np_array": list_data})

    assert np.all(model.np_array == np.array([1, 2, 3, 4.2]))


# Test dtypes handling.
def test_text_numpy_accepts_int_and_bool():
    class Model(BaseModelWithNumpy):
        np_array: np.ndarray

    model1 = Model(np_array=np.array([1, 2, 3]))
    model2 = Model(np_array=np.array([True, False, True]))

    serialized1 = model1.model_dump_json()
    serialized2 = model2.model_dump_json()

    assert json.loads(serialized1)["np_array"] == [1, 2, 3]
    assert json.loads(serialized2)["np_array"] == [True, False, True]

    deserialized1 = Model.model_validate_json(serialized1)
    deserialized2 = Model.model_validate_json(serialized2)

    assert deserialized1.np_array.dtype == np.int64
    assert np.all(deserialized1.np_array == np.array([1, 2, 3]))
    assert deserialized2.np_array.dtype == np.bool_
    assert np.all(deserialized2.np_array == np.array([True, False, True]))


@pytest.mark.parametrize(
    "np_data",
    [
        np.array(["a", "b", "c"], dtype=np.str_),
        np.array([4.0, 2.0], dtype=np.float32),
        np.array([5566, 6655], dtype=np.int16),
        np.array([{"foo": 2}, {"bar": 4}]),
    ],
)
def test_text_numpy_rejects_nonstandard_dtypes(np_data: np.ndarray):
    class Model(BaseModelWithNumpy):
        np_array: np.ndarray

    model = Model(np_array=np_data)

    with pytest.raises(ValueError, match="Cannot serialize .+ dtype"):
        model.model_dump_json()


@pytest.mark.parametrize(
    ("invalid_float", "string_value", "equality_fn"),
    [
        (np.nan, "NaN", np.isnan),
        (np.inf, "Infinity", np.isposinf),
        (-np.inf, "-Infinity", np.isneginf),
    ],
)
def test_model_dump_fixes_nan_inf(
    invalid_float: float, string_value: str, equality_fn: Callable[[float], bool]
):
    class Model(BaseModelWithNumpy):
        num: float
        num_list: list[float]
        num_array: jt.Float[np.ndarray, " n_elements"]
        num_dict: dict[str, float]

    model = Model(
        num=invalid_float,
        num_list=[0.0, invalid_float, 2.0],
        num_array=np.array([0.0, invalid_float, 2.0]),
        num_dict={"key": invalid_float},
    )

    serialized_dict = model.model_dump()
    serialized_json_dict = model.model_dump(mode="json")
    serialized_json_str = model.model_dump_json()

    # non-JSON model_dump should preserve values as floats
    assert equality_fn(serialized_dict["num"])
    assert equality_fn(serialized_dict["num_list"][1])
    assert equality_fn(serialized_dict["num_array"][1])
    assert equality_fn(serialized_dict["num_dict"]["key"])

    # JSON model_dump should convert to strings
    assert serialized_json_dict["num"] == string_value
    assert serialized_json_dict["num_list"] == [0, string_value, 2]
    assert serialized_json_dict["num_array"] == [0, string_value, 2]
    assert serialized_json_dict["num_dict"]["key"] == string_value

    # model_dump_json should convert to strings
    assert f'"num":"{string_value}"' in serialized_json_str
    assert f'"num_list":[0.0,"{string_value}",2.0]' in serialized_json_str
    assert f'"num_array":[0.0,"{string_value}",2.0]' in serialized_json_str
    assert f'"key":"{string_value}"' in serialized_json_str


def test_model_validate_accepts_nan_inf_strings():
    class Model(BaseModelWithNumpy):
        num: float
        num_list: list[float]
        num_array: jt.Float[np.ndarray, " n_elements"]
        num_dict: dict[str, float]

    serialized = {
        "num": "NaN",
        "num_list": ["NaN", "Infinity", "-Infinity"],
        "num_array": ["NaN", "Infinity", "-Infinity"],
        "num_dict": {"key0": "NaN", "key1": "Infinity", "key2": "-Infinity"},
    }

    model = Model.model_validate(serialized)

    assert np.isnan(model.num)
    assert np.isnan(model.num_list[0])
    assert np.isnan(model.num_array[0])
    assert np.isnan(model.num_dict["key0"])

    assert np.isposinf(model.num_list[1])
    assert np.isposinf(model.num_array[1])
    assert np.isposinf(model.num_dict["key1"])

    assert np.isneginf(model.num_array[2])
    assert np.isneginf(model.num_list[2])
    assert np.isneginf(model.num_dict["key2"])


def test_model_dump_mixed_dict_with_nans():
    class Inner(BaseModelWithNumpy):
        num: float

    class Outer(BaseModelWithNumpy):
        nested: dict[str, float | Inner | list[Inner]]

    model = Outer(
        nested={
            "direct": np.nan,
            "wrapped": Inner(num=np.nan),
            "wrapped_list": [Inner(num=np.nan), Inner(num=np.nan)],
        }
    )

    serialized = model.model_dump(mode="json")

    assert serialized == {
        "nested": {
            "direct": "NaN",
            "wrapped": {"num": "NaN"},
            "wrapped_list": [
                {"num": "NaN"},
                {"num": "NaN"},
            ],
        },
    }


def test_serialize_literal():
    """Regression test: Literal is supported by Pydantic, but used to be broken by
    Dapper serialization. Make sure this does not happen again."""

    class Model(BaseModelWithNumpy):
        lit_str: Literal["foo", "bar"]

    model = Model(lit_str="foo")

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert json_obj["lit_str"] == "foo"

    with pytest.raises(ValueError, match="literal_error"):
        Model.model_validate_json('{"lit_str":"invalid"}')


def test_serialization_jax_array_mixed():
    class ModelWithNpOrJaxArrays(BaseModelWithNumpy):
        untyped_array: np.ndarray | jt.Array
        typed_array: jt.Float[np.ndarray | jt.Array, " n_elements"]

    model = ModelWithNpOrJaxArrays(
        untyped_array=jnp.array([1.0, 2.0, 3.0], dtype=np.float64),
        typed_array=jnp.array([4.0, 5.0, 6.0], dtype=np.float64),
    )

    # Still JAX arrays!
    assert isinstance(model.untyped_array, jnp.ndarray)
    assert isinstance(model.typed_array, jnp.ndarray)

    serialized = model.model_dump_json()

    json_obj = json.loads(serialized)
    assert isinstance(json_obj["untyped_array"], list)
    assert json_obj["untyped_array"] == [1.0, 2.0, 3.0]
    assert isinstance(json_obj["typed_array"], list)
    assert json_obj["typed_array"] == [4.0, 5.0, 6.0]

    reconstructed = ModelWithNpOrJaxArrays.model_validate_json(serialized)

    # After validation, arrays come back as numpy!
    assert isinstance(reconstructed.untyped_array, np.ndarray)
    np.testing.assert_equal(reconstructed.untyped_array, np.array([1.0, 2.0, 3.0]))
    assert isinstance(reconstructed.typed_array, np.ndarray)
    np.testing.assert_equal(reconstructed.typed_array, np.array([4.0, 5.0, 6.0]))


def test_validation_jax_only_fails():
    """We do not allow serializing JAX-only arrays in DapperData."""

    class ModelWithOnlyJaxArrays(BaseModelWithNumpy):
        jax_only_array: jt.Array

    with pytest.raises(ValueError):  # noqa: PT011
        ModelWithOnlyJaxArrays.model_validate_json('{"jax_only_array":[1.0, 2.0, 3.0]}')
