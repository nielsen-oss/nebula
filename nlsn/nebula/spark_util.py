"""Spark utilities."""

import operator as py_operator
import sys
import warnings
from collections import Counter
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    DataType,
    DoubleType,
    FloatType,
    StructField,
    StructType,
    _parse_datatype_string,
)

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    compare_lists_of_string,
    ensure_flat_list,
    validate_regex_pattern,
)
from nlsn.nebula.logger import logger

__all__ = [
    "ALLOWED_SPARK_HASH",
    "assert_col_type",
    "cache_if_needed",
    "cast_to_schema",
    "compare_dfs",
    "drop_duplicates_no_randomness",
    "ensure_spark_condition",
    "get_column_data_type_name",
    "get_data_skew",
    "get_default_spark_partitions",
    "get_schema_as_str",
    "get_spark_condition",
    "get_spark_full_conf",
    "get_spark_session",
    "get_registered_tables",
    "hash_dataframe",
    "is_broadcast",
    "is_valid_number",
    "null_cond_to_false",
    "split_df_bool_condition",
    "string_schema_to_datatype",
    "table_is_registered",
    "take_min_max_over_window",
    "to_pandas_to_spark",
]

ALLOWED_SPARK_HASH = {"md5", "crc32", "sha1", "sha2", "xxhash64"}

ALLOWED_STANDARD_OPERATORS = {"eq", "ne", "le", "lt", "ge", "gt"}

ALLOWED_SPARK_NULL_OPERATORS = {"isNull", "isNotNull", "isNaN", "isNotNaN"}
ALLOWED_SPARK_OPERATORS = {
    "array_contains",
    "between",
    "contains",
    "isin",
    "isnotin",
    "like",
    "rlike",
    "startswith",
    "endswith",
}

_allowed_operators = ALLOWED_STANDARD_OPERATORS.union(
    ALLOWED_SPARK_NULL_OPERATORS
).union(ALLOWED_SPARK_OPERATORS)

_psql = pyspark.sql  # keep it for linting


def _string_schema_to_datatype(li: List[Iterable[str]]) -> List[StructField]:
    """Convert diamond notation schema into spark schema.

    Given a list of columns / types like:
    [
        ['col_1', 'map<string, double>'],
        ['col_2', 'string']
    ]

    return a spark schema like:
    [
        StructField(col_1, MapType(StringType, DoubleType, true), true),
        StructField(col_2, StringType, true)
    ]
    """
    li_join = []
    for el in li:
        li_join.append(" ".join(el))
    s = ", ".join(li_join)
    return list(_parse_datatype_string(s))


def assert_col_type(
    df: "pyspark.sql.DataFrame", col_name: str, col_type: Union[str, Iterable[str]]
):
    """Assert that a spark column is of the expected type."""
    dtype: str = get_column_data_type_name(df, col_name)
    if isinstance(col_type, str):
        set_col_type = {col_type}
    else:
        set_col_type = set(col_type)

    if dtype not in set_col_type:
        valid_types = " | ".join([f'"{i}"' for i in sorted(set_col_type)])
        msg = f'Input column must be "{valid_types}" type. Found "{dtype}"'
        raise TypeError(msg)


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
    cols: List[str] = df.columns

    if len(cols) != len(schema):
        logger.error("Different columns")
        names = ["dataframe", "expected"]
        exp_cols = schema.names
        diff = compare_lists_of_string(cols, exp_cols, names=names)
        print("\n".join(diff))
        msg = "'cast_subset_to_input_schema' -> Different number of columns"
        raise AssertionError(msg)

    exp_columns: List[str] = [i.name for i in schema]
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
    columns: Optional[List[str]] = None,
    raise_if_row_number_mismatch: bool = True,
    return_mismatched_rows: bool = False,
) -> Tuple[Optional["pyspark.sql.DataFrame"], Optional["pyspark.sql.DataFrame"]]:
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

    msg = (
        f"Dataframes mismatch: {n_diff_1} rows in 1st " f"df, {n_diff_2} rows in 2nd df"
    )

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
    subset: Union[str, List[str]],
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

    grouping: List[str] = ensure_flat_list(subset)

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


def ensure_spark_condition(
    operator: str,
    value: Optional[Any] = None,
    compare_col: Optional[str] = None,
) -> None:
    """Validate the input parameters for a Spark condition.

    This function checks if the provided `operator`, `value`, and `compare_col` are
    valid for constructing a Spark condition. It ensures that the operator is
    supported, that the correct arguments are provided based on the operator,
    and that the values are of the expected types.

    Args:
        operator (str):
            The comparison operator to use. Valid operators include:
            - "eq", "ne", "le", "lt", "ge", "gt" (equality, greater, lower)
            - "isNull", "isNotNull", "isNaN", "isNotNaN": look for null values
            - "isin", "isnotin": check if a value is / is not in an iterable
            - "array_contains": look for an element in a <ArrayType> Column
            - "between": looks for value between provided lower_bound and upper_bound, inclusive
            - "contains": look for a substring in a <StringType> Column
            - "startswith": look for a string that starts with.
            - "endswith": look for a string that ends with.
            - "like": values matching a pattern using _ and % in a <StringType> Column
            - "rlike": values matching a regex pattern in a <StringType> Column

        value (Any, optional):
            The value to compare against. Required for operators that compare against a
            specific value (e.g., "eq", "gt", "isin"). Cannot be used with `compare_col`.
            Defaults to None.

        compare_col (str, optional):
            The name of the column to compare against. Required for operators that compare
            two columns (e.g., "eq", "gt"). Cannot be used with `value`. Defaults to None.

    Raises:
        AssertionError:
            - If `operator` is not a string.
            - If `operator` is not in the list of allowed operators.
            - If `value` is a string when using "isin" or "isnotin".
            - If `value` is not an iterable when using "isin" or "isnotin".
            - If `None` is present in the iterable when using "isin" or "isnotin".
            - If `value` is not a list or tuple when using "between".
            - If `value` is not a string when using "contains", "startswith",
                "endswith", "like", or "rlike".
        TypeError:
            - If `value` is a string when using "isin" or "isnotin".
            - If `value` is not an iterable when using "isin" or "isnotin".
        ValueError:
            - If both `value` and `compare_col` are provided.
            - If `compare_col` is provided with operators that do not support
                column comparisons.
            - If neither `value` nor `compare_col` is provided for operators
                that require a comparison.
            - If `value` is not a list or tuple of length 2 when using "between".
            - If the regex pattern is invalid in "rlike".
    """
    assert_allowed(operator, _allowed_operators, "operator")
    if operator in ALLOWED_SPARK_NULL_OPERATORS:
        return

    if (value is not None) and (compare_col is not None):
        raise ValueError("Only one of 'value' and 'compare_col' must be provided!")

    # From now on, handle spark operator (array_contains, like, ...)
    # or standard operator (ge, eq, ...)

    if operator in ALLOWED_STANDARD_OPERATORS:
        return

    # From now on, handle spark operator (array_contains, like, ...)
    no_column_op = {"rlike", "between"}
    if (operator in no_column_op) and (compare_col is not None):
        msg = f"Column comparison is not allowed with {no_column_op} operators."
        raise ValueError(msg)

    if operator in {"isin", "isnotin"}:
        if isinstance(value, str):
            raise TypeError(
                "With 'isin' / 'isnotin' the value provided "
                "cannot be a string, use 'contains' for strings."
            )
        if not isinstance(value, Iterable):
            raise TypeError(
                "With 'isin' / 'isnotin' the value provided "
                "must be an iterable (but not a string)."
            )

        if None in value:
            raise TypeError(
                "The 'isin' / 'isnotin' operator does not handle 'None' in the iterable"
            )

    elif operator == "between":
        if not isinstance(value, (list, tuple)):
            raise ValueError('With "between" operator, value must be <list> or <tuple>')
        if len(value) != 2:
            raise ValueError("Value must be a list or tuple of length 2!")

    elif operator in {"contains", "startswith", "endswith", "like", "rlike"}:
        if not isinstance(value, str):
            raise AssertionError(f'With "{operator}" operator, the value must be <str>')

        if operator == "rlike":
            validate_regex_pattern(value)  # Raise ValueError if fails


def get_column_data_type_name(df, column: str) -> str:
    """Return the base type name of the spark dataframe column.

    Args:
        df (pyspark.sql.DataFrame):
            The input dataframe.
        column (str):
            The column from which to retrieve the data type name.

    Returns (str):
        Data type name of the specified column. I.e.:
        - "boolean"
        - "integer" (not "int")
        - "long"
        - "double"
        - "float"
        - "decimal"
        - "array"
        - "map"
    """
    return df.select(column).schema[0].dataType.typeName()


def get_spark_session(df) -> "pyspark.sql.SparkSession":
    """Retrieve the sparkSession instance from a dataframe."""
    return df.sql_ctx.sparkSession


def get_default_spark_partitions(df: "pyspark.sql.DataFrame") -> int:
    """Get the default number of spark shuffle partitions."""
    ss = get_spark_session(df)
    partitions: str = ss.conf.get("spark.sql.shuffle.partitions")
    return int(partitions)


def get_spark_full_conf(df: "pyspark.sql.DataFrame") -> List[Tuple[str, str]]:
    """Get the full spark configuration as list(tuple((str, str))."""
    ss = get_spark_session(df)
    return ss.sparkContext.getConf().getAll()


def get_registered_tables(spark: "pyspark.sql.SparkSession") -> List[str]:
    """Get the registered spark tables."""
    return [i.name for i in spark.catalog.listTables()]


def get_schema_as_str(
    df: "pyspark.sql.DataFrame", full_type_name: bool
) -> List[Tuple[str, str]]:
    """Return the dataframe schema as a List of 2-string tuples.

    The first string represents the column, the latter its data-types.

    For full_type_name=True:
    [
        ('col_1', 'string'),
        ('col_2', 'map<string, int>'),
        ('col_3', 'array<array<int>>'),
    ]

    For full_type_name=False:
    [
        ('col_1', 'string'),
        ('col_2', 'map'),
        ('col_3', 'array'),
    ]

    Args:
        df: (pyspark.sdl.Dataframe)
            Input dataframe.
        full_type_name: (bool)
            If True returns the full name for complex types, like:
                - 'map<string, map<string, int>>'
                - 'array<array<int>>'
            If False returns only the outermost data type name like:
                - 'map'
                - 'array'

    Returns: (list(tuple(str, str)))
        [(col_1, data-type name), (col_2, data-type, name), ...]
    """
    meth = "simpleString" if full_type_name else "typeName"

    fields: List[StructField] = df.schema.fields
    ret: List[Tuple[str, str]] = []
    for field in fields:
        name: str = field.name  # columns name
        data_type = field.dataType
        type_name: str = getattr(data_type, meth)()  # data type name
        ret.append((name, type_name))
    return ret


def hash_dataframe(
    df: "pyspark.sql.DataFrame",
    hash_name: str = "md5",
    *,
    new_col: Optional[str] = None,
    num_bits: int = 256,
    return_func: bool = False,
) -> Union["pyspark.sql.DataFrame", F.col]:
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

    sorted_cols: List[str] = sorted(df.columns)

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


def null_cond_to_false(cond: F.col) -> F.col:
    """Convert a null value in a boolean field to False."""
    null_cond = cond.isNull()
    return F.when(null_cond, F.lit(False)).otherwise(cond)


def get_data_skew(df, as_dict: bool = False) -> Optional[Dict[str, Any]]:
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
    li: List[int] = s_rdd.glom().map(len).collect()
    s = pd.Series(li)

    desc: List[Tuple[str, float]]
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


def get_spark_condition(
    df: "pyspark.sql.DataFrame",
    col_str: str,
    operator: str,
    *,
    value: Optional[Any] = None,
    compare_col: Optional[str] = None,
) -> F.col:
    """Verify if a condition is met.

    Args:
        df (pyspark.sql.DataFrame):
            Input dataframe.
        col_str (str):
            Column name.
        operator (str):
            Valid operators:
            - "eq", "ne", "le", "lt", "ge", "gt" (equality, greater, lower)
            - "isNull", "isNotNull", "isNaN", "isNotNaN": look for null values
            - "isin", "isnotin": check if a value is / is not in an iterable
            - "array_contains": look for an element in a <ArrayType> Column
            - "between": looks for value between provided lower_bound and upper_bound, inclusive
            - "contains": look for a substring in a <StringType> Column
            - "startswith": look for a string that starts with.
            - "endswith": look for a string that ends with.
            - "like": values matching a pattern using _ and % in a <StringType> Column
            - "rlike": values matching a regex pattern in a <StringType> Column

        value (any | None):
            Value used for the comparison.
        compare_col (str | None):
            Name of column to be compared with col_str.

        Either `value` or `compare_col` (not both) must be provided for
        python operators that require a comparison value.

    Returns (pyspark.sql.Column):
        BooleanType field to use in operation like to use in df.filter,
        df.where, F.when, ...
    """
    ensure_spark_condition(operator, value=value, compare_col=compare_col)
    spark_col = F.col(col_str)
    cond: F.col
    compare_f_col = F.col(compare_col) if compare_col else None

    if operator in ALLOWED_SPARK_NULL_OPERATORS:
        if operator == "isNull":
            cond = spark_col.isNull()
        elif operator == "isNotNull":
            cond = spark_col.isNotNull()
        elif operator == "isNaN":
            cond = F.isnan(spark_col)
        else:  # isNotNaN
            cond = ~F.isnan(spark_col)

    elif operator in ALLOWED_SPARK_OPERATORS:
        # Handle the within conditions:
        #
        # - "array_contains"
        # - "between"
        # - "contains"
        # - "isin"
        # - "isnotin"
        # - "like"
        # - "rlike"
        # - "startswith"
        # - "endswith"
        if operator in {"isin", "isnotin"}:
            if isinstance(value, Iterable):  # safer
                value = list(value)
            cond = spark_col.isin(value)
            if operator == "isnotin":
                cond = ~null_cond_to_false(cond)
        elif operator == "array_contains":
            cond = F.array_contains(spark_col, value)
        elif operator == "between":
            cond = spark_col.between(lowerBound=value[0], upperBound=value[1])
        else:
            cond = getattr(spark_col, operator)(value)

    else:  # it is in ALLOWED_STANDARD_OPERATORS
        to_compare = value if value is not None else compare_f_col
        cmp = getattr(py_operator, operator)  # comparison
        cond = cmp(spark_col, to_compare)

        # in this spark version, 3.0.0, NaN behaves in *non-intuitive* way:
        # (use the number 5 just for the example)
        # NaN > 5: True
        # NaN < 5: False
        if operator in {"le", "lt", "ge", "gt"}:
            type_c1 = get_column_data_type_name(df, col_str)
            if type_c1 in {"double", "float"}:
                cond &= ~F.isnan(spark_col)

            if compare_col is not None:
                type_c2 = get_column_data_type_name(df, compare_col)
                if type_c2 in {"double", "float"}:
                    cond &= ~F.isnan(compare_f_col)

    return cond


def is_valid_number(c: str):
    """Check if values in a specific field are valid number or not (Null /NaN).

    Args:
        c (str):
            Column name.

    Returns (pyspark.sql.Column):
        True if the value is a valid number, False if the value is Null or NaN.
    """
    return F.col(c).isNotNull() & ~F.isnan(c)


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


def split_df_bool_condition(
    df: "pyspark.sql.DataFrame",
    cond: F.col,
) -> Tuple["pyspark.sql.DataFrame", "pyspark.sql.DataFrame"]:
    """Split a dataframe into two dataframes given a certain condition.

    Given a spark condition like:
        F.col("col_1") == "my_string"
    Returns 2 dataframes, the first one is the one that meets the condition.
    The second one contains all the remaining rows, null-values in "col_1" included.

    This function handles the null-value in a different way than logical
    negation "~":
    my_cond = F.col("col_1") == "my_string"
    df1 = df.filter(my_cond)
    df2 = df.filter(~my_cond)
    Neither df1 nor df2 contains null-values since negating null-values
    returns the same null-values.

    Check the unittest for details.

    NB: this function does not handle NaN:

    df_in.show()
    +----+----+
    |  c1|  c2|
    +----+----+
    |   a| 0.0|
    |   b| 2.0|
    |   c| NaN|
    |   d|null|
    +----+----+

    df1, df2 = split_df_bool_condition(df_in, F.col("c2") > 1)

    df1.show()
    +---+---+
    | c1| c2|
    +---+---+
    |  b|2.0|
    |  c|NaN|
    +---+---+

    df2.show()
    +----+----+
    |  c1|  c2|
    +----+----+
    |   a| 0.0|
    |   d|null|
    +----+----+

    Args:
        df: (pyspark.sql.DataFrame)
            Input dataframe
        cond: (pyspark.sql.column.Column)
            Spark condition(s) E.g.:
                - F.col("col_1") == 5
                - F.col("col_1").isin(["str1", "str2"])

    Returns: (pyspark.sql.DataFrame, pyspark.sql.DataFrame)
        - DataFrame that meets the condition
        - DataFrame with remaining rows that does not meet the condition
    """
    ret_1 = df.filter(cond)
    cond_fix = null_cond_to_false(cond)
    ret_2 = df.filter(~cond_fix)

    return ret_1, ret_2


def string_schema_to_datatype(li: List[Iterable[str]]) -> List[StructField]:
    """Convert a string diamond notation schema into a spark schema.

    Given a list of columns / types like:
    [
        ['col_1', 'map<string, double>'],
        ['col_2', 'string']
    ]

    return a spark schema like:
    [
        StructField(col_1, MapType(StringType, DoubleType, true), true),
        StructField(col_2, StringType, true)
    ]
    """
    li_join = []
    el: str

    for el in li:
        li_join.append(" ".join(el))

    s: str = ", ".join(li_join)

    return list(_parse_datatype_string(s))


def table_is_registered(t: str, spark: "pyspark.sql.SparkSession") -> bool:
    """Check whether a table is registered as a Spark temporary view.

    Args:
        t (str): Table name.
        spark (SparkSession): The current spark session.

    Returns (bool):
        If the table is already registered in spark
    """
    tables = get_registered_tables(spark)
    return t in tables


def take_min_max_over_window(
    df: "pyspark.sql.DataFrame",
    windowing_columns: Union[str, List[str]],
    column: str,
    operator: str,
    perform: str,
) -> "pyspark.sql.DataFrame":
    """Window over windowing_cols and keep rows where column col_op is min/max.

    All the NaN / null-values in 'col_op' are discarded.

    E.g.
    First column = 'windowing_cols'
    Second column = 'col_op'
    [
        # a
        ["a", 0.0],  # keep for min
        ["a", 0.0],  # keep for min
        ["a", 1.0],  # never keep
        ["a", 1.0],  # never keep
        ["a", 2.0],  # keep for max
        # b
        ["b", 10.0],  # keep for min / max
        ["b", 10.0],  # keep for min / max
        ["b", 10.0],  # keep for min / max
        # c
        ["c", 11.0],  # keep for min / max
        # d
        ["d", 12.0],  # keep for min / max
        ["d", None],  # never keep
        # e
        ["e", 12.0],  # keep for min / max
        ["e", _nan],  # never keep
        # f
        ["f", None],  # never keep
        # g
        ["g", _nan],  # never keep
        # h
        ["h", 5.0],
        ["h", None],  # never keep
        ["h", _nan],  # never keep
        # None
        [None, 3.0],  # keep for min / max
        [None, _nan],  # never keep
    ]

    Args:
        df (pyspark.sql.dataframe):
            Input dataframe.
        windowing_columns (str | list(str)):
            Column(s) used to define the window partitions.
        column (str):
            Specifies the operation to perform, either "min" or "max".
        operator (str):
            "min" or "max".
        perform (str):
            Specifies the action to take, either "filter" or "replace".
            - "filter": Retains only the rows where the values are the
                computed min/max, discarding all NaN/null values in the
                input column.
            - "replace": Replaces all values that are not the computed
                min/max with the min/max value, replacing all NaN/null
                values in the input column. If a window contains only
                    None / NaN, the outcome for such a window will be None.

    Returns: (pyspark.sql.dataframe)
        Spark DataFrame with only the rows where the min/max condition is met.
    """
    assert_allowed(operator, {"min", "max"}, "operator")
    assert_allowed(perform, {"filter", "replace"}, "perform")

    fun = getattr(F, operator)
    w = Window.partitionBy(windowing_columns)

    col_op_type: DataType = df.select(column).schema[0].dataType
    nan_types = [FloatType(), DoubleType()]
    is_real_number = col_op_type in nan_types

    if is_real_number:
        c_op = F.when(F.isnan(column), F.lit(None)).otherwise(F.col(column))
    else:
        c_op = F.col(column)

    df = df.withColumn("_no_nan_", c_op).withColumn("_op_", fun(c_op).over(w))
    if perform == "replace":
        df = df.withColumn(column, F.col("_op_"))
    else:
        df = df.filter(F.col("_no_nan_") == F.col("_op_"))
    return df.drop("_op_", "_no_nan_")


def to_pandas_to_spark(df):
    """Convert a pyspark dataframe to Pandas and revert to Spark."""
    cols = df.columns
    if len(cols) != len(set(cols)):
        more = sorted([k for k, v in Counter(cols).items() if v > 1])
        msg = f"Duplicated columns, cannot execute the dataframe conversion: {more}"
        raise AssertionError(msg)
    spark_session = df.sql_ctx.sparkSession
    schema = df.schema
    return spark_session.createDataFrame(df.toPandas(), schema=schema)


def cache_if_needed(df, do_cache: bool):
    """Cache the dataframe if is not already cached."""
    if not do_cache:
        return df
    if df.is_cached:
        logger.info("DataFrame was already cached, no need to persist.")
    else:
        df = df.cache()
    return df
