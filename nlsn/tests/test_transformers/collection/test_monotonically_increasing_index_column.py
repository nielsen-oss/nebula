"""Unit-test for MonotonicallyIncreasingIndexColumn."""

import pytest

from nlsn.nebula.spark_transformers import MonotonicallyIncreasingIndexColumn

_N_ROWS: int = 10_000


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [[f"{i}"] for i in range(_N_ROWS)]
    return spark.createDataFrame(data, ["c1"]).persist()


def _assert_results(results):
    # Assert number of rows
    assert len(results) == _N_ROWS

    # Assert all numbers are int
    is_int = [isinstance(i, int) for i in results]
    assert all(is_int)

    # Assert no duplicates
    set_result = set(results)
    assert len(set_result) == _N_ROWS


@pytest.mark.parametrize("start_index", [0, 1, 10])
def test_unique_idx_sequential_true(df_input, start_index):
    """Test MonotonicallyIncreasingIndexColumn with sequential=True."""
    t = MonotonicallyIncreasingIndexColumn(
        output_col="unique_idx", sequential=True, start_index=start_index
    )
    df_chk = t.transform(df_input)
    results = df_chk.select("unique_idx").rdd.flatMap(lambda x: x).collect()
    _assert_results(results)

    li_exp = list(range(start_index, _N_ROWS + start_index))
    assert li_exp == results


def test_unique_idx_sequential_false(df_input):
    """Test MonotonicallyIncreasingIndexColumn with sequential=False."""
    t = MonotonicallyIncreasingIndexColumn(output_col="unique_idx")
    df_chk = t.transform(df_input)

    results = df_chk.select("unique_idx").rdd.flatMap(lambda x: x).collect()
    _assert_results(results)
