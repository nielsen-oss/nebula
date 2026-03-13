# Nebula - Overview

> This folder (`/ai/`) is designed as preloadable context for AI assistants working with Nebula.

## What It Is

Nebula is a Python library for building **declarative, config-driven data transformation pipelines** that run on multiple DataFrame backends: **Pandas, Polars, DuckDB, and PySpark**.

Write transformation logic once as composable pipeline steps. Run on any backend. Optionally drive everything from YAML/JSON config files.

## Core Abstractions

### 1. Transformer (base class)

Every transformation step inherits from `nebula.base.Transformer`. The base class handles backend dispatch:

```
Input DataFrame
  |
  +-- Has _transform_nw()? --> Use Narwhals (works on ALL backends)
  |
  +-- No _transform_nw()? --> Route to _transform_pandas() / _transform_polars() / _transform_spark() / _transform_duckdb()
```

- `_transform_nw(df)` — Narwhals-based, backend-agnostic. Preferred.
- `_transform_<backend>(df)` — Fallback for operations Narwhals doesn't support. Same dispatch pattern for ALL backends (not special to any one).
- The `transform(df)` method is public. It auto-wraps/unwraps between Narwhals and native formats.

### 2. TransformerPipeline

The main user-facing class (`nebula.TransformerPipeline`). Composes transformers, functions, and sub-pipelines into a reusable, inspectable pipeline.

```python
from nebula import TransformerPipeline
from nebula.transformers import SelectColumns, Filter, Cast

pipeline = TransformerPipeline([
    SelectColumns(columns=["user_id", "amount", "status"]),
    Filter(input_col="status", operator="eq", value="active"),
    Cast(cast={"amount": "float64"}),
])

result = pipeline.run(df)  # df can be pandas, polars, duckdb, or spark
```

**Capabilities:**
- **Linear pipelines** — sequential list of steps
- **Split pipelines** — partition data into named subsets, transform each, merge back
- **Branch pipelines** — fork from main or stored DF, then join/append/dead-end
- **Apply-to-rows** — filter rows by condition, transform matching subset, merge back
- **Nesting** — pipelines can contain sub-pipelines recursively
- **Storage keywords** — store/load intermediate DataFrames
- **Skip/perform** — conditional execution of pipelines or branches
- **Interleaved** — insert debug transformers between every step (dev only)
- **Config-driven** — load pipelines from YAML/JSON via `load_pipeline()`

### 3. Nebula Storage

Global in-memory key-value store (`nebula.nebula_storage`) for passing DataFrames between pipeline stages:

```python
from nebula.storage import nebula_storage as ns

ns.set("key", df)
df = ns.get("key")
```

Used for: branch inputs, intermediate checkpoints, cross-pipeline data sharing, debug snapshots.

### 4. LazyWrapper

Defers transformer instantiation until runtime, allowing parameters to depend on values computed earlier in the pipeline:

```python
from nebula.base import LazyWrapper
LazyWrapper(Filter, input_col="amount", operator="gt", value=(ns, "threshold"))
```

### 5. Pipeline Loader

`nebula.load_pipeline(config)` — builds a `TransformerPipeline` from a Python dict (loaded from YAML/JSON). Resolves transformer names, expands loops, handles lazy markers.

## Architecture

```
TransformerPipeline.__init__()
  --> IRBuilder.build()          # Build intermediate representation (tree of nodes)

TransformerPipeline.run(df)
  --> PipelineExecutor.run(df)   # Walk the IR tree, execute each node

TransformerPipeline.show()       # Text representation via PipelinePrinter
TransformerPipeline.plot()       # Graphviz DAG via GraphvizRenderer
```

## Project Structure

```
src/nebula/
  __init__.py                    # Exports: TransformerPipeline, load_pipeline, nebula_storage
  base.py                        # Transformer base class, LazyWrapper
  storage.py                     # nebula_storage singleton
  df_types.py                    # Backend type detection
  nw_util.py                     # Narwhals utilities
  backend_util.py                # Backend utilities
  auxiliaries.py                 # Column selection, schema helpers
  helpers.py                     # General helpers
  metaclasses.py                 # InitParamsStorage (auto-tracks __init__ params)
  transformers/
    __init__.py                  # Exports all transformers
    selection.py                 # SelectColumns, DropColumns, RenameColumns
    filtering.py                 # Filter, DropNulls
    combining.py                 # Join, AppendDataFrame
    reshaping.py                 # GroupBy, Pivot, Unpivot
    schema.py                    # Cast, AddLiterals
    collections.py               # When, MathOperator, DataFrameMethod, WithColumns, HorizontalFunction
    assertions.py                # AssertContainsColumns, AssertCount, AssertNotEmpty
    debug.py                     # PrintSchema
    meta.py                      # (empty/reserved)
    spark_transformers.py        # Spark-only: Repartition, Persist, Window functions, etc.
  pipelines/
    pipeline.py                  # TransformerPipeline class
    pipeline_loader.py           # load_pipeline() and config parsing
    pipe_aux.py                  # Keywords, helpers
    pipe_cfg.py                  # Configuration constants
    _checks.py                   # Input validation
    loop_expansion.py            # Loop expansion logic
    execution/
      executor.py                # PipelineExecutor
      hooks.py                   # LoggingHooks, NoOpHooks
      context.py                 # Execution context
    ir/
      builder.py                 # IRBuilder
      nodes.py                   # IR node types
      node_id.py                 # Node ID generation
    visualization/
      printer.py                 # Text output
      graphviz_renderer.py       # DAG visualization
```

## nebula_storage Lifecycle

`nebula_storage` is a **global dictionary** — it is not scoped to any pipeline. Values persist across pipeline runs until explicitly removed or the process ends. There is no automatic cleanup. It is **not thread-safe**; do not use it from multiple threads or concurrent pipelines without external synchronization.

## InitParamsStorage (Parameter Tracking)

All transformers use a metaclass (`InitParamsStorage`) that automatically records the keyword arguments passed to `__init__`. This is what powers `pipeline.show(add_params=True)` — it can display each step's configuration without transformers needing to implement any display logic. This is also why `super().__init__()` is mandatory in custom transformers: without it, parameter tracking breaks and `show()` output is incomplete.

## Dependencies

- **Core:** `narwhals >= 2.0.0` (the only required dependency)
- **Optional backends:** pandas, polars, duckdb, pyspark
- **Optional viz:** graphviz, pyyaml
