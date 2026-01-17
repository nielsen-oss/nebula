"""Unit-tests for pipeline auxiliaries."""

import os

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

from nebula.base import Transformer
from nebula.pipelines.pipe_aux import *
from nebula.pipelines.pipe_aux import PIPELINE_KEYWORDS
from nebula.transformers import *

from ...auxiliaries import from_pandas, to_pandas


class DummyTransformer(Transformer):
    """A minimal transformer for testing."""

    def __init__(self, name="unnamed"):
        super().__init__()
        self.name = name

    def _transform_nw(self, df):
        return df


class AnotherTransformer(Transformer):
    """Another transformer for testing."""

    def __init__(self):
        super().__init__()

    def _transform_nw(self, df):
        return df


def dummy_function(df):
    """A simple function for testing."""
    return df


def function_with_args(df, a, b, c=None):
    """A function with args and kwargs for testing."""
    return df


class TestIsEligibleTransformer:
    """Tests for is_eligible_transformer function."""

    def test_plain_transformer(self):
        """A plain transformer instance should be eligible."""
        trf = DummyTransformer()
        assert is_eligible_transformer(trf) is True

    def test_transformer_with_description(self):
        """A 2-tuple of (transformer, description) should be eligible."""
        trf = DummyTransformer()
        assert is_eligible_transformer((trf, "This is a description")) is True

    def test_transformer_with_empty_description(self):
        """A 2-tuple with empty string description should be eligible."""
        trf = DummyTransformer()
        assert is_eligible_transformer((trf, "")) is True

    def test_tuple_wrong_description_type(self):
        """A 2-tuple where second element is not a string should be ineligible."""
        trf = DummyTransformer()
        assert is_eligible_transformer((trf, 123)) is False
        assert is_eligible_transformer((trf, None)) is False
        assert is_eligible_transformer((trf, ["desc"])) is False

    def test_tuple_wrong_first_element(self):
        """A 2-tuple where first element is not a transformer should be ineligible."""
        assert is_eligible_transformer(("not a transformer", "desc")) is False
        assert is_eligible_transformer((dummy_function, "desc")) is False

    def test_tuple_wrong_length(self):
        """Tuples with length != 2 should be ineligible (unless plain transformer)."""
        trf = DummyTransformer()
        assert is_eligible_transformer((trf,)) is False
        assert is_eligible_transformer((trf, "desc", "extra")) is False
        assert is_eligible_transformer((trf, "desc", "extra", "more")) is False

    def test_non_transformer_types(self):
        """Non-transformer types should be ineligible."""
        assert is_eligible_transformer(None) is False
        assert is_eligible_transformer("string") is False

    def test_list_of_transformers(self):
        """A list containing transformers should be ineligible (not a single transformer)."""
        trf = DummyTransformer()
        assert is_eligible_transformer([trf]) is False
        assert is_eligible_transformer([trf, trf]) is False


class TestIsEligibleFunction:
    """Tests for is_eligible_function function."""

    def test_plain_callable(self):
        """A plain callable should be eligible."""
        assert is_eligible_function(dummy_function) is True

    def test_lambda(self):
        """A lambda should be eligible."""
        assert is_eligible_function(lambda df: df) is True

    def test_callable_class(self):
        """A callable class instance should be eligible."""

        class CallableClass:
            def __call__(self, df):
                return df

        assert is_eligible_function(CallableClass()) is True

    def test_tuple_with_args(self):
        """A 2-tuple (func, args) should be eligible."""
        assert is_eligible_function((dummy_function, [1, 2, 3])) is True
        assert is_eligible_function((dummy_function, (1, 2, 3))) is True
        assert is_eligible_function((dummy_function, [])) is True

    def test_tuple_with_args_and_kwargs(self):
        """A 3-tuple (func, args, kwargs) should be eligible."""
        assert is_eligible_function((function_with_args, [1, 2], {"c": 3})) is True
        assert is_eligible_function((function_with_args, [], {})) is True

    def test_tuple_with_args_kwargs_and_description(self):
        """A 4-tuple (func, args, kwargs, desc) should be eligible."""
        assert is_eligible_function((function_with_args, [1, 2], {"c": 3}, "description")) is True
        assert is_eligible_function((function_with_args, [], {}, "")) is True

    def test_tuple_invalid_args_type(self):
        """Second element must be list or tuple, not other types."""
        assert is_eligible_function((dummy_function, "not args")) is False
        assert is_eligible_function((dummy_function, 123)) is False
        assert is_eligible_function((dummy_function, {"a": 1})) is False
        assert is_eligible_function((dummy_function, None)) is False

    def test_tuple_invalid_kwargs_type(self):
        """Third element must be dict."""
        assert is_eligible_function((dummy_function, [], "not kwargs")) is False
        assert is_eligible_function((dummy_function, [], [1, 2])) is False
        assert is_eligible_function((dummy_function, [], None)) is False

    def test_tuple_invalid_description_type(self):
        """Fourth element must be string."""
        assert is_eligible_function((dummy_function, [], {}, 123)) is False
        assert is_eligible_function((dummy_function, [], {}, None)) is False
        assert is_eligible_function((dummy_function, [], {}, [])) is False

    def test_tuple_first_element_not_callable(self):
        """First element must be callable."""
        assert is_eligible_function(("not callable", [1, 2])) is False
        assert is_eligible_function((123, [], {})) is False
        assert is_eligible_function((None, [], {}, "desc")) is False

    def test_tuple_wrong_length(self):
        """Tuples with length < 2 or > 4 should be ineligible."""
        assert is_eligible_function((dummy_function,)) is False
        assert is_eligible_function(()) is False
        assert is_eligible_function((dummy_function, [], {}, "desc", "extra")) is False

    def test_non_callable_types(self):
        """Non-callable types should be ineligible."""
        assert is_eligible_function(None) is False
        assert is_eligible_function("string") is False
        assert is_eligible_function(123) is False
        assert is_eligible_function([]) is False
        assert is_eligible_function({}) is False

    def test_transformer_is_not_eligible_function(self):
        """Transformers should not be eligible as functions. (use is_eligible_transformer)."""
        trf = DummyTransformer()
        result = is_eligible_function(trf)
        assert result is False


class TestIsKeywordRequest:
    """Tests for is_keyword_request function."""

    def test_all_pipeline_keywords(self):
        """All defined pipeline keywords should be recognized."""
        for keyword in PIPELINE_KEYWORDS:
            assert is_keyword_request({keyword: "some_value"}) is True

    def test_unknown_keyword(self):
        """A dict with unknown key should be invalid."""
        assert is_keyword_request({"unknown_keyword": "value"}) is False
        assert is_keyword_request({"not_a_keyword": "value"}) is False

    def test_multiple_keys(self):
        """A dict with more than one key should be invalid."""
        assert is_keyword_request({"store": "a", "store_debug": "b"}) is False
        assert is_keyword_request({"store": "a", "extra": "b"}) is False

    def test_empty_dict(self):
        """An empty dict should be invalid."""
        assert is_keyword_request({}) is False

    def test_non_dict_types(self):
        """Non-dict types should be invalid."""
        assert is_keyword_request(None) is False
        assert is_keyword_request("store") is False
        assert is_keyword_request(["store", "key"]) is False
        assert is_keyword_request(("store", "key")) is False
        assert is_keyword_request({"store", "key"}) is False  # set, not dict
        assert is_keyword_request(123) is False

    def test_keyword_with_various_value_types(self):
        """Keyword requests should accept various value types."""
        if "store" in PIPELINE_KEYWORDS:
            assert is_keyword_request({"store": "string_value"}) is True
            assert is_keyword_request({"store": 123}) is True
            assert is_keyword_request({"store": None}) is True
            assert is_keyword_request({"store": ["list", "value"]}) is True


@pytest.mark.parametrize("add_params", [True, False])
@pytest.mark.parametrize("max_len", [-1, 0, 100])
@pytest.mark.parametrize("wrap_text", [True, False])
@pytest.mark.parametrize("as_list", [True, False])
def test_get_transformer_name(add_params, max_len, wrap_text, as_list):
    """Test 'get_transformer_name' function."""
    cols_select: list[str] = ["this_column_is_23_chars"] * 100
    param_len_full: int = len("".join(cols_select))
    t = SelectColumns(columns=cols_select)
    kwargs = {
        "add_params": add_params,
        "max_len": max_len,
        "wrap_text": wrap_text,
        "as_list": as_list,
    }
    if add_params and wrap_text and as_list:
        with pytest.raises(ValueError):
            get_transformer_name(t, **kwargs)
        return

    chk = get_transformer_name(t, **kwargs)

    base_len = len(f"{t.__class__.__name__} -> PARAMS: ") * 1.1  # keep a margin

    if as_list:
        assert isinstance(chk, list)
        if not add_params:
            return
        n_chk = sum(len(i) for i in chk)
        if max_len <= 0:
            assert n_chk > param_len_full, chk
        else:
            assert n_chk <= (base_len + max_len), chk
    else:
        assert isinstance(chk, str)
        if not add_params:
            return
        n_chk = len(chk)
        if max_len <= 0:
            assert n_chk > param_len_full, chk
        else:
            assert n_chk <= (base_len + max_len), chk


def _func1():
    pass


def _func2():
    pass


def _func3():
    pass


_expected_dict_extra_func = {"_func1": _func1, "_func2": _func2, "_func3": _func3}


@pytest.mark.parametrize(
    "funcs, expected",
    [
        ({}, {}),
        ([], {}),
        (_func2, {"_func2": _func2}),
        (_expected_dict_extra_func, _expected_dict_extra_func),
        ([_func1, _func2, _func3], _expected_dict_extra_func),
    ],
)
def test_create_dict_extra_functions(funcs, expected):
    """Test 'create_dict_extra_functions' function."""
    chk = create_dict_extra_functions(funcs)
    assert chk == expected


@pytest.mark.parametrize(
    "o",
    [
        "wrong",
        [_func1, _func2, _func2],
        [_func1, _func2, 1],
        {"_func1": _func1, "_func2": "_func2"},
    ],
)
def test_create_dict_extra_functions_error(o):
    """Test 'create_dict_extra_functions' function with wrong arguments."""
    with pytest.raises(AssertionError):
        create_dict_extra_functions(o)


@pytest.mark.parametrize("to_nw", [True, False])
class TestGetNativeSchema:
    def test_pandas(self, to_nw: bool):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        if to_nw:
            df = nw.from_native(df)

        sch = get_native_schema(df)

        assert isinstance(sch, dict)
        assert set(sch.keys()) == {"a", "b"}
        assert "int" in str(sch["a"]).lower()

    def test_polars(self, to_nw: bool):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        if to_nw:
            df = nw.from_native(df)

        sch = get_native_schema(df)

        assert isinstance(sch, dict)
        assert set(sch.keys()) == {"a", "b"}
        assert sch["a"] == pl.Int64
        assert sch["b"] == pl.String

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_spark(self, spark, to_nw: bool):
        df = spark.createDataFrame([("a", "x"), ("b", "y")], ["a", "b"])
        if to_nw:
            df = nw.from_native(df)

        sch = get_native_schema(df)

        assert hasattr(sch, "fields")
        field_names = [f.name for f in sch.fields]
        assert set(field_names) == {"a", "b"}
        assert isinstance(sch["a"].dataType, StringType)
        assert isinstance(sch["b"].dataType, StringType)


class TestSplitDf:
    """Test the split_df function with Polars."""

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self):
        return pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "status": ["active", "inactive", "active", "pending", "active"],
                "score": [85, 42, 91, None, 73],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            }
        )

    def test(self, df_input):
        cfg = {"input_col": "score", "operator": "gt", "value": 80}
        df_1_chk_nw, df_2_chk_nw = split_df(df_input, cfg)
        df_1_chk = nw.to_native(df_1_chk_nw)
        df_2_chk = nw.to_native(df_2_chk_nw)

        # Scores > 80: 85, 91 (2 rows)
        df_1_exp = df_input.filter(pl.col("score") > 80)
        pl.testing.assert_frame_equal(df_1_chk, df_1_exp)

        # Scores <= 80 or null: 42, None, 73 (3 rows)
        df_2_exp = df_input.filter((pl.col("score") <= 80) | pl.col("score").is_null())
        pl.testing.assert_frame_equal(df_2_chk, df_2_exp)


class TestToSchema:
    """Test suite for to_schema function."""

    @pytest.fixture(scope="class", name="list_dfs")
    def _get_list_dfs(self) -> list[pd.DataFrame]:
        df1 = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.5, 30.5], "name": ["a", "b", "c"]})

        df2 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30], "name": ["a", "b", "c"]})

        df3 = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [1.2, float("nan"), 3.1],
                "name": ["a", "b", "c"],
            }
        )

        return [df1, df2, df3]

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "n, to_nw",
        [
            (1, None),
            (1, "all"),
            (None, 1),
            (None, None),
            (None, "all"),
        ],
    )
    def test_pandas(self, backend: str, list_dfs, n, to_nw):
        dtypes = {"id": "int64", "value": "float32"}
        dataframes = list_dfs[:n]
        expected = [i.astype(dtypes) for i in dataframes]

        dataframes = [from_pandas(i, backend, to_nw=False, spark=None) for i in dataframes]

        if to_nw == 1:
            dataframes[1] = nw.from_native(dataframes[1])
        elif to_nw == "all":
            dataframes = [nw.from_native(i) for i in dataframes]

        pl_schema = {"id": pl.Int64, "value": pl.Float32}
        result = to_schema(dataframes, dtypes if backend == "pandas" else pl_schema)
        if to_nw is not None:
            result = [nw.to_native(i) for i in result]

        result = [to_pandas(i) for i in result]

        for df_chk, df_exp in zip(result, expected):
            pd.testing.assert_frame_equal(df_chk, df_exp, check_dtype=True)

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_spark(self, spark, list_dfs, to_nw):
        df_input_pd = list_dfs[0]
        df_input = from_pandas(df_input_pd, "spark", to_nw=to_nw, spark=spark)

        spark_schema = [
            StructField("id", LongType(), True),
            StructField("value", FloatType(), True),
            StructField("name", StringType(), True),
        ]

        df_chk = to_schema([df_input], StructType(spark_schema))[0]
        if to_nw:
            df_chk = nw.to_native(df_chk)

        assert df_chk.schema[1].dataType.typeName() == "float"  # in spark is 32bit
        df_exp = df_input_pd.copy().astype({"value": "float32"})
        pd.testing.assert_frame_equal(df_chk.toPandas(), df_exp, check_dtype=True)


class TestSanitizeSteps:
    """Tests for sanitize_steps function."""

    # -------------------------------------------------------------------------
    # Empty and None inputs
    # -------------------------------------------------------------------------

    def test_none_input(self):
        """None input should return empty list."""
        assert sanitize_steps(None) == []

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert sanitize_steps([]) == []

    def test_empty_tuple(self):
        """Empty tuple should return empty list."""
        assert sanitize_steps(()) == []

    # -------------------------------------------------------------------------
    # Single items (not in a list)
    # -------------------------------------------------------------------------

    def test_single_transformer(self):
        """A single transformer should be wrapped in a list."""
        trf = DummyTransformer()
        result = sanitize_steps(trf)
        assert result == [trf]

    def test_single_function(self):
        """A single function should be wrapped in a list."""
        result = sanitize_steps(dummy_function)
        assert result == [dummy_function]

    def test_single_keyword_request(self):
        """A single keyword request should be wrapped in a list."""
        keyword = list(PIPELINE_KEYWORDS)[0]  # Get first valid keyword
        request = {keyword: "test_key"}
        result = sanitize_steps(request)
        assert result == [request]

    def test_single_transformer_with_description(self):
        """A single (transformer, desc) tuple should be wrapped in a list."""
        trf = DummyTransformer()
        step = (trf, "description")
        result = sanitize_steps(step)
        assert result == [step]

    def test_single_function_with_args(self):
        """A single (func, args) tuple should be wrapped in a list."""
        step = (dummy_function, [1, 2, 3])
        result = sanitize_steps(step)
        assert result == [step]

    def test_single_function_full_tuple(self):
        """A single (func, args, kwargs, desc) tuple should be wrapped in a list."""
        step = (function_with_args, [1, 2], {"c": 3}, "description")
        result = sanitize_steps(step)
        assert result == [step]

    # -------------------------------------------------------------------------
    # Flat lists
    # -------------------------------------------------------------------------

    def test_flat_list_of_transformers(self):
        """A flat list of transformers should be returned as-is."""
        trf1 = DummyTransformer("first")
        trf2 = AnotherTransformer()
        trf3 = DummyTransformer("third")

        result = sanitize_steps([trf1, trf2, trf3])
        assert result == [trf1, trf2, trf3]

    def test_flat_list_of_functions(self):
        """A flat list of functions should be returned as-is."""
        fn1 = dummy_function
        fn2 = lambda df: df

        result = sanitize_steps([fn1, fn2])
        assert result == [fn1, fn2]

    def test_flat_list_of_keyword_requests(self):
        """A flat list of keyword requests should be returned as-is."""
        keywords = list(PIPELINE_KEYWORDS)[:2] if len(PIPELINE_KEYWORDS) >= 2 else list(PIPELINE_KEYWORDS)
        requests = [{kw: f"key_{i}"} for i, kw in enumerate(keywords)]

        result = sanitize_steps(requests)
        assert result == requests

    def test_flat_list_mixed_types(self):
        """A flat list with mixed valid types should be returned as-is."""
        trf = DummyTransformer()
        fn = dummy_function
        keyword = list(PIPELINE_KEYWORDS)[0]
        request = {keyword: "test_key"}
        trf_with_desc = (AnotherTransformer(), "description")
        fn_with_args = (function_with_args, [1], {"c": 2}, "fn desc")

        steps = [trf, fn, request, trf_with_desc, fn_with_args]
        result = sanitize_steps(steps)
        assert result == steps

    # -------------------------------------------------------------------------
    # Nested lists - flattening
    # -------------------------------------------------------------------------

    def test_nested_list_single_level(self):
        """A nested list should be flattened."""
        trf1 = DummyTransformer("first")
        trf2 = DummyTransformer("second")
        trf3 = DummyTransformer("third")

        steps = [trf1, [trf2, trf3]]
        result = sanitize_steps(steps)
        assert result == [trf1, trf2, trf3]

    def test_nested_list_multiple_levels(self):
        """Deeply nested lists should be fully flattened."""
        trf1 = DummyTransformer("1")
        trf2 = DummyTransformer("2")
        trf3 = DummyTransformer("3")
        trf4 = DummyTransformer("4")

        steps = [trf1, [trf2, [trf3, [trf4]]]]
        result = sanitize_steps(steps)
        assert result == [trf1, trf2, trf3, trf4]

    def test_nested_list_with_mixed_types(self):
        """Nested lists with mixed types should be flattened correctly."""
        trf = DummyTransformer()
        fn = dummy_function
        keyword = list(PIPELINE_KEYWORDS)[0]
        request = {keyword: "nested_key"}

        steps = [
            trf,
            [fn, request],
            [(AnotherTransformer(), "desc")],
        ]
        result = sanitize_steps(steps)

        assert len(result) == 4
        assert result[0] is trf
        assert result[1] is fn
        assert result[2] == request
        assert isinstance(result[3], tuple)

    def test_nested_tuple_as_container(self):
        """A tuple that is not a valid step should be treated as a container."""
        trf1 = DummyTransformer("1")
        trf2 = DummyTransformer("2")
        trf3 = DummyTransformer("3")

        # A 3-tuple of transformers is not a valid step, so it's a container
        steps = [trf1, (trf2, trf3)]  # This is ambiguous - depends on implementation

        # If (trf2, trf3) is not valid as transformer tuple or function tuple,
        # it should be flattened
        result = sanitize_steps(steps)
        assert trf1 in result
        assert trf2 in result
        assert trf3 in result

    def test_nested_empty_lists(self):
        """Empty nested lists should not affect the result."""
        trf = DummyTransformer()

        steps = [trf, [], [[]], [[], []]]
        result = sanitize_steps(steps)
        assert result == [trf]

    def test_all_empty_nested_lists(self):
        """Only empty nested lists should return empty list."""
        steps = [[], [[]], [[], [[]]]]
        result = sanitize_steps(steps)
        assert result == []

    # -------------------------------------------------------------------------
    # Complex real-world scenarios
    # -------------------------------------------------------------------------

    def test_realistic_pipeline(self):
        """Test a realistic pipeline configuration."""
        trf1 = DummyTransformer("drop_columns")
        trf2 = DummyTransformer("select_columns")
        trf3 = DummyTransformer("assert_not_empty")

        keyword = list(PIPELINE_KEYWORDS)[0]

        steps = [
            trf1,
            trf2,
            (trf3, "Ensure the DF is not empty"),
            dummy_function,
            (function_with_args, [1, 2, 3], {"c": 10}, "random function"),
            {keyword: "intermediate_result"},
        ]

        result = sanitize_steps(steps)

        assert len(result) == 6
        assert result[0] is trf1
        assert result[1] is trf2
        assert result[2] == (trf3, "Ensure the DF is not empty")
        assert result[3] is dummy_function
        assert result[4] == (
            function_with_args,
            [1, 2, 3],
            {"c": 10},
            "random function",
        )
        assert result[5] == {keyword: "intermediate_result"}

    def test_realistic_pipeline_with_nested_groups(self):
        """Test a pipeline with logically grouped nested steps."""
        preprocessing = [
            DummyTransformer("drop_nulls"),
            DummyTransformer("fill_na"),
        ]

        feature_engineering = [
            DummyTransformer("add_features"),
            (DummyTransformer("normalize"), "Normalize numeric columns"),
        ]

        validation = [
            (AnotherTransformer(), "Final validation"),
        ]

        steps = [
            preprocessing,
            feature_engineering,
            dummy_function,
            validation,
        ]

        result = sanitize_steps(steps)

        assert len(result) == 6
        # All nested items should be flattened in order
        assert result[0].name == "drop_nulls"
        assert result[1].name == "fill_na"
        assert result[2].name == "add_features"
        assert isinstance(result[3], tuple)  # (normalize, desc)
        assert result[4] is dummy_function
        assert isinstance(result[5], tuple)  # (validation, desc)

    # -------------------------------------------------------------------------
    # Order preservation
    # -------------------------------------------------------------------------

    def test_order_preserved_flat(self):
        """Order should be preserved in flat lists."""
        transformers = [DummyTransformer(str(i)) for i in range(10)]
        result = sanitize_steps(transformers)

        for i, trf in enumerate(result):
            assert trf.name == str(i)

    def test_order_preserved_nested(self):
        """Order should be preserved when flattening nested lists."""
        t0 = DummyTransformer("0")
        t1 = DummyTransformer("1")
        t2 = DummyTransformer("2")
        t3 = DummyTransformer("3")
        t4 = DummyTransformer("4")

        steps = [t0, [t1, t2], t3, [t4]]
        result = sanitize_steps(steps)

        assert [t.name for t in result] == ["0", "1", "2", "3", "4"]

    # -------------------------------------------------------------------------
    # Error cases
    # -------------------------------------------------------------------------

    def test_invalid_type_string(self):
        """A string should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps("not a valid step")

    def test_invalid_type_integer(self):
        """An integer should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps(123)

    def test_invalid_type_in_list(self):
        """An invalid type inside a list should raise TypeError."""
        trf = DummyTransformer()
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps([trf, "invalid", trf])

    def test_invalid_type_in_nested_list(self):
        """An invalid type in a nested list should raise TypeError."""
        trf = DummyTransformer()
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps([trf, [trf, 123]])

    def test_invalid_dict_not_keyword(self):
        """A dict that is not a keyword request should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps({"not_a_keyword": "value"})

    def test_invalid_dict_multiple_keys(self):
        """A dict with multiple keys should raise TypeError."""
        keywords = list(PIPELINE_KEYWORDS)[:2]
        if len(keywords) >= 2:
            with pytest.raises(TypeError, match="Invalid step"):
                sanitize_steps({keywords[0]: "a", keywords[1]: "b"})

    def test_invalid_tuple_in_list(self):
        """An invalid tuple format should raise TypeError."""
        # A tuple that doesn't match any valid pattern
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps([("not_callable", [1, 2, 3])])

    def test_set_is_invalid(self):
        """A set should raise TypeError (common mistake for dict)."""
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps({"store", "key"})  # set, not dict

    def test_set_in_list_is_invalid(self):
        """A set inside a list should raise TypeError."""
        trf = DummyTransformer()
        with pytest.raises(TypeError, match="Invalid step"):
            sanitize_steps([trf, {"store", "key"}])

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_transformer_tuple_not_flattened(self):
        """A valid (transformer, desc) tuple should NOT be flattened."""
        trf = DummyTransformer()
        step = (trf, "description")

        result = sanitize_steps([step])

        assert len(result) == 1
        assert result[0] == step
        assert isinstance(result[0], tuple)

    def test_function_tuple_not_flattened(self):
        """A valid function tuple should NOT be flattened."""
        step = (function_with_args, [1, 2], {"c": 3}, "desc")

        result = sanitize_steps([step])

        assert len(result) == 1
        assert result[0] == step

    def test_lambda_functions(self):
        """Lambda functions should be valid."""
        fn1 = lambda df: df
        fn2 = lambda df: df.filter(True)

        result = sanitize_steps([fn1, fn2])
        assert result == [fn1, fn2]

    def test_callable_class_instance(self):
        """Callable class instances should be valid as functions."""

        class CallableProcessor:
            def __call__(self, df):
                return df

        processor = CallableProcessor()
        result = sanitize_steps([processor])
        assert result == [processor]

    def test_very_deep_nesting(self):
        """Very deeply nested structures should be handled."""
        trf = DummyTransformer()

        # Create 10 levels of nesting
        nested = trf
        for _ in range(10):
            nested = [nested]

        result = sanitize_steps(nested)
        assert result == [trf]

    def test_multiple_keyword_requests_in_sequence(self):
        """Multiple keyword requests in sequence should work."""
        keyword = list(PIPELINE_KEYWORDS)[0]
        requests = [
            {keyword: "first"},
            {keyword: "second"},
            {keyword: "third"},
        ]

        result = sanitize_steps(requests)
        assert result == requests
