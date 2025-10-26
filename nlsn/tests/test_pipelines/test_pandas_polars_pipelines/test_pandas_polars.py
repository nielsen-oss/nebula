"""Test pandas pipelines."""

# This module is awful, it must be refactored

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.base import Transformer
from nlsn.nebula.pandas_polars_transformers import Count
from nlsn.nebula.pipelines._pandas_split_functions import pandas_split_function
from nlsn.nebula.pipelines._polars_split_functions import polars_split_function
from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.shared_transformers import (
    Distinct,
    PrintSchema,
    RoundValues,
    WithColumn,
)
from nlsn.nebula.spark_transformers import SelectColumns
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.auxiliaries import assert_pandas_polars_frame_equal, pandas_to_polars
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml

from .._shared import RuntimeErrorTransformer
from . import _custom_extra_transformers
from .auxiliaries import align_and_assert, get_input_pandas_df

_BACKENDS = ["pandas", "polars"]


def _load_and_run_pipe(data, backend: str, df_input, pipe_kws) -> pd.DataFrame:
    data_backend = deepcopy(data)
    data_backend["backend"] = backend
    pipe = load_pipeline(data_backend, **pipe_kws)
    pipe.show_pipeline(add_transformer_params=True)

    df = df_input.copy()
    df = pandas_to_polars(backend, df)

    df_out = pipe.run(df)
    if backend == "polars":
        return df_out.to_pandas()
    return df_out


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return get_input_pandas_df()


class TestExplictBackend:
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self):
        """Get input dataframe."""
        data = [
            [0, "a0", "b0"],
            [1, "a1", "b1"],
            [2, "a2", "b2"],
        ]
        return pd.DataFrame(data, columns=["idx", "c1", "c2"])

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_explicit_backend(
        self,
        df_input,
        backend: str,
    ):
        """Test explicit and implicit pandas backend."""
        pipe = TransformerPipeline(SelectColumns(glob="c*"), backend=backend)
        pipe.show_pipeline()

        df_exp = df_input[["c1", "c2"]]
        df_chk = pipe.run(pandas_to_polars(backend, df_input))
        align_and_assert(backend, df_exp, df_chk)


class TestExtraTransformers:
    """Test extra transformers."""

    @staticmethod
    def _get_add_one() -> type:
        class AddOne(Transformer):
            backends = {"pandas", "polars"}

            def __init__(self, *, column_name: str):
                """Wrong transformer, it will be superseded by the one in the module."""
                super().__init__()
                self._name: str = column_name

            def _transform(self, df):
                return self._select_transform(df)

            def _transform_pandas(self, df):
                ret = df.copy()
                ret[self._name] += 10  # Keep 10 (wrong value)
                return ret

            def _transform_polars(self, df):
                # Add 10 (wrong value) instead of 1, so the assertions can
                # check whether the original AddOne has been superseded
                # by this one
                value = pl.col(self._name) + 10
                return df.with_columns(value.alias(self._name))

        return AddOne

    @staticmethod
    def _get_with_column_2() -> type:
        class WithColumnExtra(Transformer):
            backends = {"pandas", "polars"}

            def __init__(self, *, column_name, value):
                """Pandas WithColumn transformer."""
                super().__init__()
                self._name: str = column_name
                self._value = value

            def _transform(self, df):
                return self._select_transform(df)

            def _transform_pandas(self, df):
                ret = df.copy()
                ret[self._name] = self._value
                return ret

            def _transform_polars(self, df):
                value = pl.lit(self._value)
                return df.with_columns(value.alias(self._name))

        return WithColumnExtra

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_extra_transformers(self, backend: str):
        """Test extra transformers."""
        df_input = pd.DataFrame([[0, "a"]], columns=["idx", "c1"])

        df_exp = df_input.copy()
        df_exp["idx"] += 1
        df_exp["c_x"] = 10
        df_exp["c_y"] = "Y"

        data = {
            "pipeline": [
                {
                    "transformer": "WithColumn",
                    "params": {"column_name": "c_x", "value": 10},
                },
                {
                    "transformer": "WithColumnExtra",
                    "params": {"column_name": "c_y", "value": "Y"},
                },
                {
                    "transformer": "AddOne",
                    "params": {"column_name": "idx"},
                },
            ]
        }

        @dataclass
        class ExtraTransformers:
            AddOne = self._get_add_one()
            WithColumnExtra = self._get_with_column_2()

        pipe_kws = {
            "extra_transformers": [
                _custom_extra_transformers,
                ExtraTransformers,
            ],
        }
        df_chk = _load_and_run_pipe(data, backend, df_input, pipe_kws)
        pd.testing.assert_frame_equal(df_exp, df_chk, check_dtype=False)


class TestPipelineLoader:
    @staticmethod
    def _get_pipe_kws(backend: str):
        if backend == "pandas":
            func = pandas_split_function
        else:
            func = polars_split_function

        def _split_func_outer(df):
            df_low, df_hi = func(
                df,
                input_col="idx",
                operator="lt",
                value=10,
                compare_col=None,
            )
            return {
                "outer_low": df_low,
                "outer_hi": df_hi,
            }

        def _split_func_inner(df):
            df_low, df_hi = func(
                df,
                input_col="idx",
                operator="lt",
                value=5,
                compare_col=None,
            )
            return {
                "inner_low": df_low,
                "inner_hi": df_hi,
            }

        return {
            "extra_transformers": [_custom_extra_transformers],
            "extra_functions": {
                "split_func_outer": _split_func_outer,
                "split_func_inner": _split_func_inner,
            },
        }

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("pipeline_key", ["split-is-none", "split-is-empty-list"])
    def test_mock_pipelines_empty_splits(
        self, df_input, backend: str, pipeline_key: str
    ):
        """Test split-pipelines with an empty split.

        The transformers in this pipeline do nothing, just count.

        Check if the pipeline return the same dataframe.
        """
        fname = "empty_split.yml"
        data = load_yaml(fname)[pipeline_key]

        df_chk = _load_and_run_pipe(
            data, backend, df_input, self._get_pipe_kws(backend)
        )
        align_and_assert(backend, df_chk, df_input.copy(), force_sort=True)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_pipeline_loader_splits_allow_missing_columns(self, df_input, backend: str):
        """Test split-pipelines with an allow_missing_columns=True.

        Check if the pipeline return the same dataframe.
        """
        fname = "split_allow_missing_columns.yml"
        data = load_yaml(fname)["split-allow-missing-columns"]

        pipe_kws = self._get_pipe_kws(backend)

        df_chk = _load_and_run_pipe(data, backend, df_input, pipe_kws)

        pd_split_func_outer = self._get_pipe_kws("pandas")["extra_functions"][
            "split_func_outer"
        ]
        dict_df = pd_split_func_outer(df_input)
        df_low = dict_df["outer_low"].copy()
        df_low["new_column"] = "new_value"
        df_exp = pd.concat([dict_df["outer_hi"], df_low], axis=0)

        align_and_assert(backend, df_chk, df_exp)


class TestFlatPipeline:
    _TRANSFORMERS = [SelectColumns(glob="*"), Distinct()]
    _INTERLEAVED = PrintSchema()

    @pytest.fixture(scope="class", name="df_exp")
    def _get_df_exp(self, df_input):
        """Get expected dataframe."""
        df_ret = df_input.copy()
        for t in self._TRANSFORMERS:
            df_ret = t.transform(df_ret)
        return df_ret

    @staticmethod
    def _eval(df_input, df_exp, backend, pipe):
        df_input = pandas_to_polars(backend, df_input)
        df_exp = pandas_to_polars(backend, df_exp)
        pipe.show_pipeline()
        df_chk = pipe.run(df_input)
        assert_pandas_polars_frame_equal(backend, df_chk, df_exp, check_row_order=False)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize(
        "interleaved, prepend_interleaved, append_interleaved, name",
        [
            (None, True, False, None),
            ([], True, False, None),
            (_INTERLEAVED, True, False, "name_01"),
            (_INTERLEAVED, True, True, "name_02"),
            (_INTERLEAVED, False, False, "name_03"),
        ],
    )
    def test_pipeline_flat_list_transformers(
        self,
        df_input,
        df_exp,
        backend: str,
        interleaved: list,
        prepend_interleaved: bool,
        append_interleaved: bool,
        name: str,
    ):
        """Test TransformerPipeline pipeline w/ list of transformers."""
        pipe = TransformerPipeline(
            self._TRANSFORMERS,
            interleaved=interleaved,
            prepend_interleaved=prepend_interleaved,
            append_interleaved=append_interleaved,
            name=name,
            backend=backend,
        )
        self._eval(df_input, df_exp, backend, pipe)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("split_func", [None, lambda x: x])
    def test_pipeline_single_split(self, df_input, backend: str, df_exp, split_func):
        """Test TransformerPipeline pipeline w/ list of transformers."""
        pipe = TransformerPipeline(
            {"no split": self._TRANSFORMERS},
            split_function=split_func,
            name="single split",
        )
        self._eval(df_input, df_exp, backend, pipe)


class TestSplitPipeline:
    _INTERLEAVED = PrintSchema()

    _APPLY_BEFORE_APPENDING = Distinct()

    _trf_low: list = [RoundValues(input_columns="idx", precision=3)]
    _trf_hi: list = [RoundValues(input_columns="idx", precision=1)]

    @staticmethod
    def _get_split_function(backend: str, value=10):
        if backend == "pandas":
            func = pandas_split_function
        else:
            func = polars_split_function

        def _split_func(df):
            df_low, df_hi = func(
                df.copy() if isinstance(df, pd.DataFrame) else df,
                input_col="idx",
                operator="lt",
                value=value,
                compare_col=None,
            )
            return {
                "low": df_low,
                "hi": df_hi,
            }

        return _split_func

    def _get_df_exp(self, df_input, split_order):
        """Get expected dataframe."""
        msk = df_input["idx"] < 10
        df_low = df_input[msk].copy()
        df_hi = df_input[~msk].copy()

        for t in self._trf_low + [self._APPLY_BEFORE_APPENDING]:
            df_low = t.transform(df_low)

        for t in self._trf_hi + [self._APPLY_BEFORE_APPENDING]:
            df_hi = t.transform(df_hi)

        li_df = [df_hi, df_low]
        if split_order:
            if split_order[0] == "low":
                li_df = li_df[::-1]

        df_ret = pd.concat(li_df, axis=0)
        # "interleaved" transformers are just for log in this unit-test.
        return df_ret

    @pytest.mark.parametrize("keys", [("no", "hi"), ("low", "hi", "no")])
    def test_wrong_keys(self, df_input, keys):
        """Test TransformerPipeline pipeline with wrong keys in dict_splits."""
        dict_splits = {i: Count() for i in keys}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=self._get_split_function("pandas"),
            name="test_error",
        )

        with pytest.raises(KeyError):
            pipe.run(df_input)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize(
        "name, split_order, interleaved, prepend_interleaved, append_interleaved, cast_subset_to_input_schema",
        [
            (None, None, None, True, False, False),
            (None, [], [], True, False, True),
            ("name_01", None, _INTERLEAVED, True, False, False),
            ("name_02", ["low", "hi"], _INTERLEAVED, True, True, True),
            ("name_03", ["hi", "low"], _INTERLEAVED, True, True, True),
        ],
    )
    def test_split_pipeline(
        self,
        df_input,
        backend: str,
        name: str,
        split_order,
        interleaved: list,
        prepend_interleaved: bool,
        append_interleaved: bool,
        cast_subset_to_input_schema,
    ):
        """Test TransformerPipeline pipeline."""
        dict_splits = {"low": self._trf_low, "hi": self._trf_hi}

        pipe = TransformerPipeline(
            dict_splits,
            backend=backend,
            split_function=self._get_split_function(backend),
            name=name,
            split_order=split_order,
            interleaved=interleaved,
            prepend_interleaved=prepend_interleaved,
            append_interleaved=append_interleaved,
            split_apply_before_appending=self._APPLY_BEFORE_APPENDING,
            cast_subset_to_input_schema=cast_subset_to_input_schema,
        )

        df_exp = self._get_df_exp(df_input.copy(), split_order)

        pipe.show_pipeline()
        pipe._print_dag()
        df_input = df_input.copy()
        df_input = pandas_to_polars(backend, df_input)
        df_chk = pipe.run(df_input)

        if cast_subset_to_input_schema:
            df_exp["idx"] = df_exp["idx"].astype("float64")

        align_and_assert(backend, df_chk, df_exp)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_allow_missing_columns(self, df_input, backend: str):
        """Test TransformerPipeline pipeline with allow_missing_columns=True."""
        t_new_col = WithColumn(column_name="new_column", value="new_value")
        dict_splits = {"low": [], "hi": [t_new_col]}

        pipe = TransformerPipeline(
            dict_splits,
            backend=backend,
            split_function=self._get_split_function(backend),
            allow_missing_columns=True,
        )

        pd_split_func = self._get_split_function("pandas")
        dict_df = pd_split_func(df_input)
        df_low = dict_df["low"].copy()
        df_hi = dict_df["hi"].copy()
        df_hi["new_column"] = "new_value"
        df_exp = pd.concat([df_hi, df_low], axis=0)

        if backend == "polars":
            df_input = pl.from_pandas(df_input)
        else:
            df_input = df_input.copy()

        df_chk = pipe.run(df_input)

        align_and_assert(backend, df_chk, df_exp)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize(
        "splits_skip_if_empty", [None, "hi", {"hi"}, {"hi", "low"}]
    )
    def test_splits_skip_if_empty(self, df_input, backend: str, splits_skip_if_empty):
        """Test TransformerPipeline pipeline with splits_skip_if_empty argument.

        The highest value in the dataframe for this test is 15, therefore the
        split "hi" receives an empty df, and the "low" split is without
        any transformation, returning the fully untouched input dataframe.
        """
        dict_splits = {"low": [], "hi": [RuntimeErrorTransformer()]}

        pipe = TransformerPipeline(
            dict_splits,
            backend=backend,
            split_function=self._get_split_function(backend, value=100),
            splits_skip_if_empty=splits_skip_if_empty,
        )

        if backend == "polars":
            df_input = pl.from_pandas(df_input)
        else:
            df_input = df_input.copy()

        if splits_skip_if_empty in {None, "low"}:
            with pytest.raises(RuntimeError):
                pipe.run(df_input)
            return

        df_chk = pipe.run(df_input)
        align_and_assert(backend, df_chk, df_input)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("splits_skip_if_empty", ["wrong", {"wrong"}, ["wrong"]])
    def test_splits_skip_if_empty_wrong_split(self, backend: str, splits_skip_if_empty):
        """Test TransformerPipeline pipeline with splits_skip_if_empty and a wrong split name."""
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"low": [], "hi": []},
                backend=backend,
                split_function=self._get_split_function(backend),
                splits_skip_if_empty=splits_skip_if_empty,
            )


class TestStorageFunctionalities:
    """Test some storage functionalities."""

    @staticmethod
    @pytest.mark.parametrize("source", ["python", "yaml"])
    def test_replace_with_stored_df(df_input, source: str):
        """Test 'replace_with_stored_df' functionality."""
        ns.clear()
        df2 = pd.concat([df_input.copy(), df_input.copy()], axis=0)
        store_data = {"replace_with_stored_df": "df2"}

        if source == "python":
            pipe = TransformerPipeline([store_data])
        else:
            pipe = load_pipeline({"pipeline": store_data})

        pipe.show_pipeline()
        pipe._print_dag()

        try:
            ns.set("df2", df2)
            df_out = pipe.run(df_input)
            pd.testing.assert_frame_equal(df_out, df2)
        finally:
            ns.clear()
