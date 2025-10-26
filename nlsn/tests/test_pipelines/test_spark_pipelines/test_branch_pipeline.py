"""Test 'branch' pipeline functionalities."""

import os

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.test_pipelines._shared import DICT_BRANCH_PIPELINE
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml

# Available pipelines:
# - branch_dead_end_without_storage: tested with pandas & polars
# - branch_append_without_storage: tested with pandas & polars
# - test_branch_append_without_storage_error: keep for the spark error
# - branch_join_without_storage: tested with pandas & polars
# - branch_dead_end_with_storage: tested with pandas & polars
# - branch_append_with_storage: tested with pandas & polars
# - branch_append_with_storage_error: keep for the spark error
# - branch_append_with_storage_new_col: tested with pandas and polars
# - branch_join_with_storage: tested with pandas & polars
# - repartition_to_original
# - coalesce_to_original
# - branch_append_otherwise: tested with pandas and polars
# - branch_join_otherwise: tested with pandas and polars
# - branch_skip: tested with pandas and polars
# - branch_skip_otherwise: tested with pandas and polars


_SOURCES = ["py", "yaml"]


def __create_df(spark):
    # Called in pytest.fixture and as a standalone function.
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]

    data = [
        [0, "a", "b"],
        [0, "a", "b"],  # dropped with Distinct()
        [0, "a", "b"],  # dropped with Distinct()
        [1, "a", "b"],
        [2, "a", "b"],
        [3, "", ""],
        [4, "", ""],
        [5, None, None],
        [6, " ", None],
        [7, "", None],
        [8, "a", None],
        [9, "a", ""],
        [10, "", "b"],
        [11, "a", None],
        [12, None, "b"],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).cache()


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    return __create_df(spark)


def _assert_dfs(df_chk, df_exp):
    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_nullable=True)


def _get_pipe(name: str, source: str):
    if source == "yaml":
        yaml_date = load_yaml("branch.yml")
        return load_pipeline(yaml_date[name])
    elif source == "py":
        return DICT_BRANCH_PIPELINE[name]()
    else:
        raise RuntimeError


def _run(pipe_name, source, df_input):
    pipeline = _get_pipe(pipe_name, source)
    pipeline.show_pipeline()
    return pipeline.run(df_input)


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_dead_end_without_storage(df_input, source):
    """Test 'branch' & 'dead-end' without 'storage'."""
    ns.clear()

    pipe_name = "branch_dead_end_without_storage"
    df_out = _run(pipe_name, source, df_input)
    df_exp = df_input.distinct()
    _assert_dfs(df_out, df_exp)

    df_chk_fork = ns.get("df_fork")
    df_exp_fork = df_input.distinct().withColumn("new", F.lit(-1))
    _assert_dfs(df_chk_fork, df_exp_fork)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_without_storage(df_input, source):
    """Test 'branch' & 'append' without 'storage'."""
    ns.clear()

    pipe_name = "branch_append_without_storage"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_distinct.withColumn("c1", F.lit("c"))

    df_exp = df_distinct.unionByName(df_fork)
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_without_storage_error(df_input, source):
    """Test 'branch' & 'append' without 'storage'."""
    ns.clear()

    pipe_name = "branch_append_without_storage_error"
    pipeline = _get_pipe(pipe_name, source)
    pipeline.show_pipeline()

    with pytest.raises(AnalysisException):
        pipeline.run(df_input)

    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_without_storage_new_col(df_input, source):
    """Test 'branch' & 'append' without 'storage'."""
    ns.clear()

    pipe_name = "branch_append_without_storage_new_col"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_distinct.withColumn("new_column", F.lit("new_value"))

    df_exp = df_distinct.unionByName(df_fork, allowMissingColumns=True)
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_join_without_storage(df_input, source):
    """Test 'branch' & 'join' without 'storage'."""
    ns.clear()

    pipe_name = "branch_join_without_storage"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_distinct.select("idx").withColumn("new", F.lit(-1))

    df_exp = df_distinct.join(df_fork, on="idx", how="inner")
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_dead_end_with_storage(df_input, source):
    """Test 'branch' & 'dead-end' with 'storage'."""
    ns.clear()
    ns.set("df_x", df_input)

    pipe_name = "branch_dead_end_with_storage"
    df_out = _run(pipe_name, source, df_input)
    df_exp = df_input.distinct()
    _assert_dfs(df_out, df_exp)

    df_chk_fork = ns.get("df_fork")
    df_exp_fork = df_input.withColumn("new", F.lit(-1))
    _assert_dfs(df_chk_fork, df_exp_fork)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_with_storage(df_input, source):
    """Test 'branch' & 'append' with 'storage'."""
    ns.clear()
    ns.set("df_x", df_input)

    pipe_name = "branch_append_with_storage"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_input.withColumn("c1", F.lit("c"))

    df_exp = df_distinct.unionByName(df_fork)
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_with_storage_error(df_input, source):
    """Test 'branch' & 'append' with 'storage' but not allowed new column."""
    ns.clear()
    ns.set("df_x", df_input)

    pipe_name = "branch_append_with_storage_error"
    pipeline = _get_pipe(pipe_name, source)
    pipeline.show_pipeline()

    with pytest.raises(AnalysisException):
        pipeline.run(df_input)

    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_with_storage_new_col(df_input, source):
    """Test 'branch' & 'append' with 'storage' and a new column."""
    ns.clear()
    ns.set("df_x", df_input)

    pipe_name = "branch_append_with_storage_new_col"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_input.withColumn("new_column", F.lit("new_value"))

    df_exp = df_distinct.unionByName(df_fork, allowMissingColumns=True)
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_join_with_storage(spark, df_input, source):
    """Test 'branch' & 'join' with 'storage'."""
    ns.clear()

    # Recreate the dataframe, do not pass 'df_input' as it
    # will be joined with itself.
    df_secondary = __create_df(spark)
    ns.set("df_x", df_secondary)

    pipe_name = "branch_join_with_storage"
    df_out = _run(pipe_name, source, df_input)

    df_distinct = df_input.distinct()
    df_new = df_secondary.select("idx").withColumn("new", F.lit(-1))

    # here I set broadcast = True
    df_exp = df_distinct.join(F.broadcast(df_new), on="idx", how="inner")
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.parametrize("name", ["repartition_to_original", "coalesce_to_original"])
def test_coalesce_repartition_to_original(df_input, name: str):
    """Test 'branch' & 'join' with 'storage'."""
    ns.clear()

    n_exp = df_input.rdd.getNumPartitions()

    pipeline = _get_pipe(name, "yaml")
    pipeline.show_pipeline()

    df_out = pipeline.run(df_input)
    n_chk = df_out.rdd.getNumPartitions()
    assert n_chk == n_exp

    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_otherwise(df_input, source):
    """Test 'branch' & 'append' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_append_otherwise"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_distinct.withColumn("c1", F.lit("c"))

    df_otherwise = df_distinct.withColumn("c1", F.lit("other"))

    df_exp = df_otherwise.unionByName(df_fork)
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_otherwise(df_input, source):
    """Test 'branch' & 'join' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_join_otherwise"
    df_out = _run(pipe_name, source, df_input)
    df_distinct = df_input.distinct()
    df_fork = df_distinct.select("idx").withColumn("new", F.lit(-1))

    df_otherwise = df_distinct.withColumn("other_col", F.lit("other"))

    df_exp = df_otherwise.join(df_fork, on="idx", how="inner")
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_skip(df_input, source):
    """Test 'branch' & 'skip' pipeline."""
    ns.clear()

    pipe_name = "branch_skip"
    df_out = _run(pipe_name, source, df_input)

    df_exp = df_input.distinct()
    _assert_dfs(df_out, df_exp)
    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_skip_otherwise(df_input, source):
    """Test 'branch' & 'skip' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_skip_otherwise"
    df_out = _run(pipe_name, source, df_input)

    df_exp = df_input.distinct().withColumn("c1", F.lit("other"))
    _assert_dfs(df_out, df_exp)
    ns.clear()
