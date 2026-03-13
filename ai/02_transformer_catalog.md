# Nebula - Transformer Catalog

All transformers inherit from `nebula.base.Transformer`. All `__init__` parameters are **keyword-only** (after `*`).

Import: `from nebula.transformers import <TransformerName>`

---

## Selection

### SelectColumns
Select a subset of columns.
```python
SelectColumns(
    *,
    columns: str | list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
)
```
Backend: Narwhals. Exactly one selector must be provided.

### DropColumns
Drop columns by name/pattern.
```python
DropColumns(
    *,
    columns: str | list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
    allow_excess_columns: bool = True,
)
```
Backend: Narwhals. `allow_excess_columns=True` means no error if columns don't exist.

### RenameColumns
Rename columns.
```python
RenameColumns(
    *,
    columns: str | list[str] | None = None,          # original names
    columns_renamed: str | list[str] | None = None,   # new names (paired with columns)
    mapping: dict[str, str] | None = None,             # {old: new}
    regex_pattern: str | None = None,                  # regex to match
    regex_replacement: str | None = None,              # replacement string
    fail_on_missing_columns: bool = True,
)
```
Backend: Narwhals. Use either columns+columns_renamed, mapping, or regex_pattern+regex_replacement.

---

## Filtering

### Filter
Filter rows by condition.
```python
Filter(
    *,
    input_col: str,
    perform: str = "keep",        # "keep" or "drop"
    operator: str,                 # see operators list below
    value=None,                    # comparison value
    compare_col: str | None = None,  # compare with another column instead of value
)
```
Backend: Narwhals.

**Operators:** `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `isin`, `between`, `contains`, `startswith`, `endswith`, `like`, `rlike`, `array_contains`, `is_null`, `is_not_null`, `is_nan`, `is_not_nan`

- `is_null`, `is_not_null`, `is_nan`, `is_not_nan` do not require `value`.
- `between` expects `value` as `[lower, upper]` (inclusive).
- `isin` expects `value` as a list.
- `compare_col` and `value` are mutually exclusive.

### DropNulls
Drop rows with null values.
```python
DropNulls(
    *,
    how: str = "any",             # "any" or "all"
    thresh: int | None = None,    # minimum non-null count to keep row
    drop_na: bool = False,        # also drop NaN (not just null)
    columns: str | list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
)
```
Backend: backend-specific (pandas/polars/spark/duckdb).

---

## Combining

### Join
Join with a DataFrame from nebula storage.
```python
Join(
    *,
    store_key: str,                # key in nebula_storage
    how: str,                      # "inner", "left", "right", "full", "cross", "semi", "anti"
    on: list[str] | str | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    suffix: str = "_right",
    broadcast: bool = False,       # Spark only: broadcast hint
    coalesce_keys: bool = True,    # merge duplicate key columns
)
```
Backend: Narwhals.

### AppendDataFrame
Vertically concatenate with a DataFrame from storage. The stored DF is appended **after** the main pipeline DF (i.e., `[main_df, stored_df]`).
```python
AppendDataFrame(
    *,
    store_key: str | None = None,
    allow_missing_cols: bool = False,  # fill missing columns with null
    relax: bool = False,
    rechunk: bool = False,
    ignore_index: bool = False,
)
```
Backend: custom (uses helper functions). When `allow_missing_cols=False` (default), schema mismatches raise an error.

---

## Reshaping

### GroupBy
Group and aggregate.
```python
GroupBy(
    *,
    aggregations: dict[str, list[str]] | dict[str, str] | list[dict[str, str]],
    groupby_columns: str | list[str] | None = None,
    groupby_regex: str | None = None,
    groupby_glob: str | None = None,
    groupby_startswith: str | Iterable[str] | None = None,
    groupby_endswith: str | Iterable[str] | None = None,
    prefix: str = "",
    suffix: str = "",
)
```
Backend: Narwhals.

**Aggregation formats:**
- `{"sum": ["col_1", "col_2"]}` — single aggregation on multiple columns (supports `prefix`/`suffix`)
- `{"sum": "col_1"}` — single aggregation on single column
- `[{"agg": "sum", "col": "dollars", "alias": "total"}]` — explicit list of dicts. Keys `agg` and `col` are mandatory, `alias` is optional (defaults to `col` name). `prefix`/`suffix` are always applied on top of the final column name (whether that's `alias` or the default `col` name), following Polars/Narwhals naming behavior.

**Supported aggregations:** `sum`, `mean`, `min`, `max`, `count`, `first`, `last`, `std`, `var`, `median`, `len`, `n_unique`

### Pivot
Long to wide.
```python
Pivot(
    *,
    pivot_col: str,
    id_cols: str | list[str] | None = None,    # group columns (or id_regex, id_glob, etc.)
    id_regex: str | None = None,
    id_glob: str | None = None,
    id_startswith: str | Iterable[str] | None = None,
    id_endswith: str | Iterable[str] | None = None,
    aggregate_function: Literal["min","max","first","last","sum","mean","median","len"] = "first",
    values_cols: str | list[str] | None = None,  # value columns (or values_regex, etc.)
    values_regex: str | None = None,
    values_glob: str | None = None,
    values_startswith: str | Iterable[str] | None = None,
    values_endswith: str | Iterable[str] | None = None,
    separator: str = "_",
)
```
Backend: Narwhals.

### Unpivot (Melt)
Wide to long.
```python
Unpivot(
    *,
    id_cols: str | list[str] | None = None,
    id_regex: str | None = None,
    melt_cols: str | list[str] | None = None,
    melt_regex: str | None = None,
    variable_col: str,
    value_col: str,
)
```
Backend: Narwhals. Also exported as `Melt` (alias).

---

## Schema & Types

### Cast
Cast columns to specified types.
```python
Cast(*, cast: dict[str, str])
# Example: Cast(cast={"amount": "float64", "id": "int32"})
```
Backend: Narwhals (with Spark/Polars fallbacks for complex types).

**Common type strings:** `int8`, `int16`, `int32`, `int64`, `float32`, `float64`, `string`, `boolean`, `date`, `datetime`

### AddLiterals
Add constant-value columns.
```python
AddLiterals(*, data: list[dict])
# Example: AddLiterals(data=[
#     {"alias": "status", "value": "active"},
#     {"alias": "version", "value": 2, "cast": "int32"},
# ])
```
Backend: Narwhals. Each dict has keys: `alias` (column name), `value`, and optional `cast`.

---

## Collections (Column Operations)

### When
Conditional column creation (if-then-else).
```python
When(
    *,
    output_col: str,
    conditions: list[dict[str, Any]],
    otherwise_constant: Any = None,
    otherwise_col: str | None = None,
    cast_output: str | None = None,
)
```
Backend: Narwhals.

Each condition dict: `{"input_col": ..., "operator": ..., "value": ..., "then_constant": ... or "then_col": ...}`
Uses same operators as Filter.

### MathOperator
Chain math operations on columns.
```python
MathOperator(*, strategy: dict | list[dict])
```
Backend: Narwhals.

Strategy dict: `{"input_col": ..., "operator": "add"|"sub"|"mul"|"div"|"pow"|"mod", "value": ..., "output_col": ...}`
- `value` can be a number or column name (prefix with `col:` for column reference).

### DataFrameMethod
Call any Narwhals DataFrame method dynamically.
```python
DataFrameMethod(
    *,
    method: str,                    # e.g. "sort", "unique", "head"
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
)
```
Backend: Narwhals. Escape hatch for operations not covered by specific transformers.

### WithColumns
Apply a column method to multiple columns.
```python
WithColumns(
    *,
    columns: str | list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
    method: str,                    # Narwhals Expr method name
    alias: str | None = None,      # single column rename
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    prefix: str | None = None,     # prefix for output column names
    suffix: str | None = None,     # suffix for output column names
)
```
Backend: Narwhals. Applies `nw.col(col).method(*args, **kwargs)` to each selected column.

### HorizontalFunction
Apply Narwhals horizontal functions across columns.
```python
HorizontalFunction(
    *,
    output_col: str,
    function: str,                  # "concat_str", "max_horizontal", "min_horizontal", "sum_horizontal", "mean_horizontal", "all_horizontal", "any_horizontal", "coalesce"
    columns: list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
)
```
Backend: Narwhals.

---

## Assertions (Data Quality)

### AssertContainsColumns
Raise `AssertionError` if columns are missing.
```python
AssertContainsColumns(*, columns: str | list[str])
```
Backend: Narwhals.

### AssertCount
Assert row count constraints.
```python
AssertCount(
    *,
    expected: int | None = None,    # exact count
    min_count: int | None = None,
    max_count: int | None = None,
)
```
Backend: backend-specific.

### AssertNotEmpty
Raise `AssertionError` if DataFrame is empty.
```python
AssertNotEmpty(*, df_name: str = "DataFrame")
```
Backend: custom.

---

## Debug

### PrintSchema
Log DataFrame column types.
```python
PrintSchema()  # no parameters
```
Backend: Narwhals.

---

## Spark-Only Transformers

These only work with PySpark DataFrames. Import from `nebula.transformers.spark_transformers` or `nebula.transformers`.

### Repartition
```python
Repartition(
    *,
    num_partitions: int | None = None,
    rows_per_partition: int | None = None,
    columns: str | list[str] | None = None,
)
```

### CoalescePartitions
```python
CoalescePartitions(
    *,
    num_partitions: int | None = None,
    rows_per_partition: int | None = None,
)
```

### Persist / Cache
```python
Persist()  # no parameters
Cache()    # alias for Persist
```

### AggregateOverWindow
Window aggregation.
```python
AggregateOverWindow(
    *,
    partition_cols: str | list[str] | None = None,
    aggregations: list[dict[str, str]] | dict[str, str],
    order_cols: str | list[str] | None = None,
    ascending: bool | list[bool] = True,
    rows_between: tuple[str | int, str | int] = None,
    range_between: tuple[str | int, str | int] = None,
)
```
Aggregation dict: `{"column": ..., "agg": "sum"|"avg"|"min"|"max"|"count", "alias": ...}`

### LagOverWindow
Lag window function.
```python
LagOverWindow(
    *,
    partition_cols: str | list[str] | None = None,
    order_cols: str | list[str] | None = None,
    lag_col: str,
    lag: int,
    output_col: str,
    ascending: bool | list[bool] = True,
    rows_between: tuple[str | int, str | int] = None,
    range_between: tuple[str | int, str | int] = None,
)
```

### ColumnsToMap
Create a MapType column from multiple columns.
```python
ColumnsToMap(
    *,
    output_column: str,
    columns: str | list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
    cast_values: str | None = None,
    drop_input_columns: bool = False,
)
```

### MapToColumns
Extract MapType keys into columns.
```python
MapToColumns(
    *,
    input_column: str,
    output_columns: list[str] | list[list[str]] | dict[Any, str],
)
```

### SparkSqlFunction
Call any `pyspark.sql.functions` function.
```python
SparkSqlFunction(
    *,
    column: str,
    function: str,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
)
```

### SparkColumnMethod
Call any `pyspark.sql.Column` method.
```python
SparkColumnMethod(
    *,
    input_column: str,
    output_column: str | None = None,
    method: str,
    args: list[Any] = None,
    kwargs: dict[str, Any] | None = None,
)
```

### SparkDropDuplicates
Drop duplicate rows.
```python
SparkDropDuplicates(
    *,
    columns: str | list[str] | None = None,
    regex: str | None = None,
    glob: str | None = None,
    startswith: str | Iterable[str] | None = None,
    endswith: str | Iterable[str] | None = None,
)
```

### SparkExplode
Explode array/map column.
```python
SparkExplode(
    *,
    input_col: str,
    output_cols: str | list[str] | None = None,
    outer: bool = True,
    drop_after: bool = False,
)
```

### LogDataSkew
Log partition distribution stats.
```python
LogDataSkew(*, persist: bool = False)
```

### CpuInfo
Log CPU info of Spark workers.
```python
CpuInfo(*, n_partitions: int = 100)
```
