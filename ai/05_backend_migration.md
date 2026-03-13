# Nebula - Backend Migration Guide

## Supported Backends

| Backend | Type | Lazy Support | Notes |
|---------|------|-------------|-------|
| Pandas  | Eager | No | Most common starting point |
| Polars  | Eager + Lazy | Yes (`to_lazy`/`collect`) | High performance |
| DuckDB  | Lazy (relation) | Yes (always lazy) | SQL-friendly |
| PySpark | Lazy (by default) | Yes | Distributed |

---

## What Works Across All Backends (Narwhals)

All transformers with `_transform_nw` work on every backend without changes:

- **Selection:** SelectColumns, DropColumns, RenameColumns
- **Filtering:** Filter
- **Combining:** Join
- **Reshaping:** GroupBy, Pivot, Unpivot
- **Schema:** Cast, AddLiterals
- **Collections:** When, MathOperator, DataFrameMethod, WithColumns, HorizontalFunction
- **Assertions:** AssertContainsColumns
- **Debug:** PrintSchema

These are the ~16 transformers that are fully portable.

---

## Spark -> Polars Migration

### Transformers to Remove

These are Spark-only and have no equivalent in Polars/Pandas/DuckDB:

| Spark Transformer | Action | Rationale |
|-------------------|--------|-----------|
| `Repartition` | Remove | No partitioning concept |
| `CoalescePartitions` | Remove | No partitioning concept |
| `Persist` / `Cache` | Remove | No lazy caching needed |
| `LogDataSkew` | Remove | No partitions to analyze |
| `CpuInfo` | Remove | Spark worker diagnostics |
| `SparkExplode` | Replace | Use Polars `.explode()` via `DataFrameMethod` |
| `SparkDropDuplicates` | Replace | Use `DataFrameMethod(method="unique")` |
| `SparkSqlFunction` | Replace | Use `WithColumns` or `DataFrameMethod` |
| `SparkColumnMethod` | Replace | Use `WithColumns` |
| `AggregateOverWindow` | Replace | Not yet in Narwhals; use native Polars |
| `LagOverWindow` | Replace | Not yet in Narwhals; use native Polars |
| `ColumnsToMap` | Replace | Use Polars struct/dict operations |
| `MapToColumns` | Replace | Use Polars struct/dict operations |

### Transformers That Need Attention

| Transformer | Notes |
|-------------|-------|
| `DropNulls` | Works on both but uses backend-specific code |
| `AssertCount` | Works on both but uses backend-specific code |
| `Cast` | Mostly works via Narwhals; complex nested types may need backend-specific handling |

### Pipeline Options to Remove/Adjust

| Option | Action |
|--------|--------|
| `repartition_output_to_original` | Remove (Spark-only) |
| `coalesce_output_to_original` | Remove (Spark-only) |
| `broadcast` in branch dict | Remove (Spark-only, ignored by other backends) |

### Polars-Specific Opportunities

- Use `"to_lazy"` / `"collect"` keywords for query optimization
- Polars is eager by default but lazy mode can improve performance
- No need for `Persist`/`Cache` — Polars handles memory differently

---

## Spark -> DuckDB Migration

Similar to Spark -> Polars, but note:

- DuckDB relations are always lazy (like Spark)
- DuckDB uses SQL under the hood — custom transformers can use SQL strings via `_transform_duckdb`
- `.union()` is used instead of `append` for concatenation
- `allow_missing_columns=True` works via SQL NULL insertion

---

## Pandas -> Polars Migration

Generally the easiest migration:

- Most Narwhals transformers work identically
- Main difference: Polars is stricter about types (no implicit casting)
- `Cast` may need to be added where Pandas was lenient
- Index-based operations don't exist in Polars

---

## Writing Backend-Portable Custom Transformers

**Preferred approach:** Use `_transform_nw` with Narwhals API.

```python
from nebula.base import Transformer
import narwhals as nw

class MyTransformer(Transformer):
    def __init__(self, *, column: str, multiplier: float):
        super().__init__()
        self.column = column
        self.multiplier = multiplier

    def _transform_nw(self, df):
        return df.with_columns(
            (nw.col(self.column) * self.multiplier).alias(f"{self.column}_scaled")
        )
```

**When Narwhals isn't enough:** Implement backend-specific methods.

```python
class ComplexTransformer(Transformer):
    def __init__(self, *, column: str):
        super().__init__()
        self.column = column

    def _transform_polars(self, df):
        import polars as pl
        return df.with_columns(pl.col(self.column).over("group").alias("result"))

    def _transform_spark(self, df):
        from pyspark.sql import functions as F, Window
        w = Window.partitionBy("group")
        return df.withColumn("result", F.col(self.column).over(w))
```

**Hybrid approach:** Use `to_native`/`from_native` keywords in pipeline.

```python
pipe = TransformerPipeline([
    NarwhalsTransformer(...),       # works on any backend
    "to_native",                     # switch to native
    native_only_function,            # uses backend-specific API
    "from_native",                   # switch back to Narwhals
    AnotherNarwhalsTransformer(...),
])
```
