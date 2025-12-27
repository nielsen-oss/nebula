"""Spark utilities."""

import sys
import warnings
from io import StringIO
from typing import Any, Union

import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql.types import StructType

from nebula.auxiliaries import (
    assert_allowed,
    compare_lists_of_string,
    ensure_flat_list,
)
from nebula.logger import logger

__all__ = [
    "ALLOWED_SPARK_HASH",
    "cast_to_schema",
    "compare_dfs",
    "drop_duplicates_no_randomness",
    "get_data_skew",
    "get_default_spark_partitions",
    "get_spark_session",
    "hash_dataframe",
    "is_broadcast",
]

ALLOWED_SPARK_HASH = {"md5", "crc32", "sha1", "sha2", "xxhash64"}

_psql = pyspark.sql  # keep it for linting


def cast_to_schema(
    df: "pyspark.sql.DataFrame",
    schema: StructType,
) -> "pyspark.sql.DataFrame":
    """Cast the columns of a DataFrame to a given schema.

    Args:
        df (pyspark.sql.DataFrame):
            The input DataFrame to be cast
        schema (pyspark.sql.types.StructType):
            The target schema to which the DataFrame columns should be cast.

    Returns (pyspark.sql.DataFrame):
        A new DataFrame with columns cast to match the specified schema.

    Raises:
        AssertionError:
            If the number of columns in the DataFrame does not match
            the number of fields in the schema.
    """
    cols: list[str] = df.columns

    if len(cols) != len(schema):
        logger.error("Different columns")
        names = ["dataframe", "expected"]
        exp_cols = schema.names
        diff = compare_lists_of_string(cols, exp_cols, names=names)
        print("\n".join(diff))
        msg = "'cast_subsets_to_input_schema' -> Different number of columns"
        raise AssertionError(msg)

    exp_columns: list[str] = [i.name for i in schema]
    df = df.select(exp_columns)

    new_cols = []
    name: str
    for field in schema:
        name = field.name
        chk_type = df.select(name).schema[0].dataType
        exp_type = field.dataType
        if chk_type != exp_type:
            msg = "Different type in '{}': check={} -> expected={}"
            logger.info(msg.format(name, chk_type, exp_type))
            new_cols.append(F.col(name).cast(exp_type).alias(name))
        else:
            new_cols.append(name)
    return df.select(new_cols)


def compare_dfs(
    df1: "pyspark.sql.DataFrame",
    df2: "pyspark.sql.DataFrame",
    *,
    columns: list[str] = None,
    raise_if_row_number_mismatch: bool = True,
    return_mismatched_rows: bool = False,
) -> tuple[pyspark.sql.DataFrame | None, pyspark.sql.DataFrame | None]:
    """Compare two DFs for equality based on schema, row count, and content.

    This function performs several checks for equality. First, if no specific
    columns are provided, it verifies that both dataframes have the same set
    of columns. It then checks for schema match based on the selected or
    common columns. Next, it verifies if they have the same number of rows.
    Finally, it hashes the data in the selected columns of each row, sorts the
    resulting hash arrays, and compares the sorted arrays to determine
    if the data content is identical, regardless of row order.

    Args:
        df1 (DataFrame):
            The first PySpark DataFrame to compare.
        df2 (DataFrame):
            The second PySpark DataFrame to compare.
        columns (list(str) | None):
            A list of column names to include in the comparison.
            If None, all columns in df1 are used, provided that df2 also
            contains the exact same set of column names. An error is raised
            otherwise. Defaults to None.
        raise_if_row_number_mismatch (bool):
            If True, raises an AssertionError if the number of rows differs
            between the two DataFrames. If False, prints a warning and continues
            to compare content. Defaults to True.
        return_mismatched_rows (bool):
            If True, returns DataFrames containing the rows found in one DataFrame
            but not the other (based on content hash) when a content mismatch is
            detected. If False, raises an AssertionError on content mismatch.
            Defaults to False.

    Returns (tuple(DataFrame | None, DataFrame | None)):
        A tuple of two Optional DataFrames.
        If the Dataframes are determined to be equal (all checks pass),
        returns (None, None).
        If `return_mismatched_rows` is True and content mismatches are found,
        returns a tuple where the first DataFrame contains rows from df1 not
        present in df2 (based on hash), and the second DataFrame contains rows
        from df2 not present in df1.

    Raises:
        AssertionError:
            If `columns` is None and the sets of column names in `df1` and `df2` differ.
            If the schemas do not match (based on selected/common columns).
            If the number of rows differs and `raise_if_row_number_mismatch` is True.
            If the hashed and sorted data content of the dataframes is not equal
            and `return_mismatched_rows` is False.
    """
    # FIXME: in helpers?
    if columns is None:
        if set(df1.columns) != set(df2.columns):
            raise AssertionError("Different columns in the dataframes!")
        print("Columns are not provided, using the 1st df columns as reference")
        columns = df1.columns
        df2 = df2.select(*columns)
    else:
        df1 = df1.select(*columns)
        df2 = df2.select(*columns)

    print(f"Selected {len(columns)} columns for the comparison")

    sch_1: str = df1.schema.simpleString()
    sch_2: str = df2.schema.simpleString()
    assert sch_1 == sch_2, f"Schemas mismatch:\n1){sch_1}\n\n2)\n{sch_2}"
    print("Schemas match!\nAsserting number of rows, counting the 1st df ...")

    n1 = df1.count()
    print(f"{n1} rows. Counting the 2nd df ...")
    n2 = df2.count()

    number_mismatch: bool = n1 != n2
    if number_mismatch:
        if raise_if_row_number_mismatch:
            raise AssertionError(f"Number of rows mismatch: {n1} != {n2}")
        print(f"!!! Number of rows mismatch: {n1} != {n2} !!!")
    else:
        print("Number of rows matches!\nHashing the dataframes ...")

    print("Hashing the 1st df ...")
    df1_hash = hash_dataframe(df1, "md5", new_col="hashed")
    s1 = df1_hash.select("hashed").toPandas()["hashed"]

    print("Hashing the 2nd df ...")
    df2_hash = hash_dataframe(df2, "md5", new_col="hashed")
    s2 = df2_hash.select("hashed").toPandas()["hashed"]

    print("Sorting the hash array of the 1st df ...")
    ar_hash_1 = sorted(s1.tolist())

    print("Sorting the hash array of the 2nd df ...")
    ar_hash_2 = sorted(s2.tolist())

    print("Final assertion ...")
    hash_match: bool = ar_hash_1 == ar_hash_2
    if hash_match:
        print("Dataframes match!")
        return None, None

    set_1 = set(s1)
    set_2 = set(s2)
    set_diff_1 = set_1.difference(set_2)
    set_diff_2 = set_2.difference(set_1)

    n_diff_1 = len(set_diff_1)
    n_diff_2 = len(set_diff_2)

    msg = f"Dataframes mismatch: {n_diff_1} rows in 1st df, {n_diff_2} rows in 2nd df"

    if not return_mismatched_rows:
        raise AssertionError(msg)

    print(msg)

    # Find rows in df1_hash not in df2_hash based on hash
    df1_diff = df1_hash.join(df2_hash.select("hashed"), on=["hashed"], how="left_anti")

    # Find rows in df2_hash not in df1_hash based on hash
    df2_diff = df2_hash.join(df1_hash.select("hashed"), on=["hashed"], how="left_anti")

    return df1_diff.drop("hashed"), df2_diff.drop("hashed")


def drop_duplicates_no_randomness(
    df: "pyspark.sql.DataFrame",
    subset: str | list[str],
    agg_func: str = "max",
) -> "pyspark.sql.DataFrame":
    """Drop duplicated rows considering a subset of columns and remove the randomness when possible.

    Spark method drop_duplicates with argument subset != None can lead
    to randomness as it's akin to df.groupby(subset).first(remaining columns).
    This function applies a groupby operation followed by a min/max
    aggregation if the column type is sortable.

     This function must be used only with a valid "subset" argument.
     If the str / list is empty it raises an error. Use .distinct() or plain
     .drop_duplicates() if the transformation involves all the columns.

    In the returned dataframe:
    - Null values are not chosen if valid ones are available for the group.
    - If the remaining columns are sortable (i.e., no MapType, ArrayType),
        there is no randomness in the output dataframe.
    - Otherwise, if there are non-sortable types in the remaining columns, the
        "first" method is applied as aggregation and randomness can occur.

    Args:
        df (pyspark.sql.DataFrame):
            Input dataframe.
        subset (str | list(str)):
            The 'subset' argument of DataFrame.drop_duplicates
            If provided an empty str / list, it raises an error.
        agg_func (str):
            "min" or "max".

    Returns: (pyspark.sql.DataFrame)
        Spark DataFrame without duplicates.
    """
    if not subset or len(subset) == 0:
        raise AssertionError("subset cannot be empty")

    if isinstance(subset, str) and (len(subset.strip()) == 0):
        raise AssertionError("Cannot provide empty string as subset")

    assert_allowed(agg_func, {"min", "max"}, "agg_func")

    def _get_typename(_df, _c: str) -> str:
        return _df.select(_c).schema[0].dataType.typeName()

    def _is_scalar(_df, _c: str) -> bool:
        # False if the column is MapType or ArrayType, otherwise True
        _tp_name = _get_typename(_df, _c)
        return _tp_name not in {"map", "array"}

    func: callable = getattr(F, agg_func)

    grouping: list[str] = ensure_flat_list(subset)

    other_cols = [c for c in df.columns if c not in grouping]
    scalar_cols = [c for c in other_cols if _is_scalar(df, c)]
    non_scalar_cols = [c for c in other_cols if not _is_scalar(df, c)]

    if non_scalar_cols:
        msg = "Unable to sort the following complex columns:\n"
        list_msg = [f"{c}: {_get_typename(df, c)}" for c in non_scalar_cols]
        msg += "\n".join(list_msg)
        msg += "\nThese types are not sortable and could lead to randomness"
        warnings.warn(msg)

    list_agg_scalar = [func(c).alias(c) for c in scalar_cols]
    list_agg_non_scalar = [F.first(c).alias(c) for c in non_scalar_cols]
    list_agg = list_agg_scalar + list_agg_non_scalar

    return df.groupby(subset).agg(*list_agg)


def get_spark_session(df) -> "pyspark.sql.SparkSession":
    """Retrieve the sparkSession instance from a dataframe."""
    return df.sql_ctx.sparkSession


def get_default_spark_partitions(df: "pyspark.sql.DataFrame") -> int:
    """Get the default number of spark shuffle partitions."""
    ss = get_spark_session(df)
    partitions: str = ss.conf.get("spark.sql.shuffle.partitions")
    return int(partitions)


def hash_dataframe(
    df: "pyspark.sql.DataFrame",
    hash_name: str = "md5",
    *,
    new_col: str | None = None,
    num_bits: int = 256,
    return_func: bool = False,
) -> Union["pyspark.sql.DataFrame", "F.col"]:
    """Hash each dataframe row.

    All the columns are sorted before being hashed to ensure a repeatable result.

    Valid 'hash_name' function:
    - "md5"
    - "crc32"
    - "sha1"
    - "sha2"
    - "xxhash64"

    F.hash is not a hash function; it just returns the row hash number
    used by spark internally.

    Args:
        df (pyspark.sql.DataFrame):
            The input dataframe.
        new_col (str | None):
            Name of the new column to store the hash values.
            Defaults to None
        hash_name (str):
            Hash function name, allowed values: "md5", "crc32", "sha1",
            "sha2", "xxhash64". Defaults to "md5".
        num_bits (int):
            Number of bits for the SHA-2 hash.
            Permitted values: 0, 224, 256, 384, 512,
            Ignored if hash_name is not "sha2". Defaults to 256.
        return_func (bool):
            If True returns the hashing function, otherwise return a dataframe
            with the new column containing the hash values. Defaults to False

    Returns (pyspark.sql.DataFrame | F.col):
        A new dataframe with the additional hash column if 'return_func' is
        False, the new hashed column only otherwise.
    """
    assert_allowed(hash_name, ALLOWED_SPARK_HASH, "hash_name")
    if new_col and return_func:
        raise AssertionError("Only one among 'new_col' and 'return_func' is allowed")

    sorted_cols: list[str] = sorted(df.columns)

    hash_func = getattr(F, hash_name)

    if hash_name == "xxhash64":
        hashed_col = hash_func(*sorted_cols)
    else:
        # The following hashing functions need a single column.
        # Concatenate the columns in a single using 'concat_ws'.
        string_cols = [F.col(i).cast("string") for i in sorted_cols]
        concatenated = F.concat_ws("-", *string_cols)

        if hash_name == "sha2":
            hashed_col = F.sha2(concatenated, num_bits)
        else:
            hashed_col = hash_func(concatenated)

    if return_func:
        return hashed_col

    return df.withColumn(new_col, hashed_col)


def get_data_skew(df, as_dict: bool = False) -> dict[str, Any] | None:
    """Get the skewness of a spark dataframe.

    Args:
        df: (pyspark.sql.Dataframe)
        as_dict: (bool)

    Returns:
        If as_dict = True returns a 3-element dict like:
        {
            'partitions': 1,
             'skew': 'mean=10.00 | std=nan | min=10.00 | 25%=10.00 | 50%=10.00 | 75%=10.00 | max=10.00',
             'full_distribution': [10]
        }
        Where the key "partitions" represents the number of partitions and
        "full_distribution" the list of number of rows per partition.

        If as_dict = False (default) returns None and prints something like:
            Number of partitions: 1
            Skewness: mean=10.00 | std=nan | min=10.00 | 25%=10.00 | 50%=10.00 | 75%=10.00 | max=10.00
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        msg = "'pandas' optional package not installed. \n"
        msg += "Run 'pip install pandas' or 'install nebula[pandas]'"
        raise ImportError(msg) from exc

    col_0: str = list(df.columns)[0]  # faster conversion to rdd
    s_rdd = df.select(col_0).rdd
    n_part: int = s_rdd.getNumPartitions()

    # get the length of each partition
    li: list[int] = s_rdd.glom().map(len).collect()
    s = pd.Series(li)

    desc: list[tuple[str, float]]
    desc = list(s.describe().to_frame().to_records())
    # something ordered like
    # [('count', 10.0),
    #  ('mean', 4.5),
    #  ('std', 3.0276503540974917),
    #  ('min', 0.0),
    #  ('25%', 2.25),
    #  ('50%', 4.5),
    #  ('75%', 6.75),
    #  ('max', 9.0)]

    useless = {"count"}  # count => same as numPartitions
    d_skew = {str(k).strip().lower(): v for k, v in desc if k not in useless}

    # just to have them sorted
    keys = ["mean", "std", "min", "25%", "50%", "75%", "max"]

    if set(keys).issubset(d_skew):
        li_msg = [f"{k}={d_skew[k]:,.1f}" for k in keys]
    else:  # pragma: no cover
        li_msg = [f"{k}={v:,.1f}" for k, v in desc if k not in d_skew.items()]

    msg = " | ".join(li_msg)

    if as_dict:
        return {"partitions": n_part, "skew": msg, "full_distribution": li}
    else:  # pragma: no cover
        print(f"Number of partitions: {n_part}")
        print(f"Skewness: {msg}")
        return None


def is_broadcast(df) -> bool:
    """Check whether a dataframe is broadcast."""
    # Capture the output of '.explain'
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    df.explain(True)

    sys.stdout = old_stdout

    # Check if "Broadcast" is in the captured output
    explain_output = my_stdout.getvalue()
    return "broadcast" in explain_output.lower()
