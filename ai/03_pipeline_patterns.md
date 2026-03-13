# Nebula - Pipeline Patterns

All patterns use `from nebula import TransformerPipeline`.

---

## 1. Flat (Linear) Pipeline

Sequential list of transformers.

```python
pipe = TransformerPipeline([
    SelectColumns(columns=["id", "amount", "status"]),
    Filter(input_col="status", operator="eq", value="active"),
    Cast(cast={"amount": "float64"}),
])
result = pipe.run(df)
```

---

## 2. Using Bare Functions

Regular Python functions can be pipeline steps. The function receives the DataFrame as first argument and must return a DataFrame.

```python
def my_function(df):
    return df

pipe = TransformerPipeline([
    SelectColumns(columns=["a", "b"]),
    my_function,                                    # bare function
])
```

### Functions with Arguments (Tuple Format)

```python
def fn_with_args(df, x, y, z=10):
    return df

pipe = TransformerPipeline([
    (fn_with_args, [1, 2], {"z": 99}, "description"),   # (callable, args, kwargs, description)
    (fn_with_args, [1, 2], {"z": 99}),                  # (callable, args, kwargs) - no description
    (fn_with_args, [1, 2]),                              # (callable, args) - no kwargs
    (my_function,),                                      # (callable,) - just wrapping
])
```

### Transformer with Description

```python
pipe = TransformerPipeline([
    (AssertNotEmpty(), "Ensure the DF is not empty"),   # (transformer_instance, description)
])
```

---

## 3. Split Pipeline

Partition data into named subsets, apply different transformers to each, merge results back.

```python
def split_by_value(df):
    return {
        "high": df.filter(pl.col("amount") >= 100),
        "low": df.filter(pl.col("amount") < 100),
    }

pipe = TransformerPipeline(
    {
        "high": [AddLiterals(data=[{"alias": "tier", "value": "premium"}])],
        "low": [AddLiterals(data=[{"alias": "tier", "value": "standard"}])],
    },
    split_function=split_by_value,
)
result = pipe.run(df)
```

**Options:**
- `split_order=["low", "high"]` — control execution order (default: alphabetical)
- `split_apply_after_splitting=[AssertNotEmpty()]` — applied to each subset right after splitting
- `split_apply_before_appending=[DataFrameMethod(method="unique")]` — applied to each subset before merging back
- `splits_no_merge=["audit"]` — dead-end splits, excluded from merged output
- `splits_skip_if_empty=["rare_case"]` — skip processing if subset is empty
- `cast_subsets_to_input_schema=True` — ensure schema consistency before merge
- `allow_missing_columns=True` — fill missing columns with null during merge
- `repartition_output_to_original=True` — (Spark) restore original partition count
- `coalesce_output_to_original=True` — (Spark) coalesce instead of repartition

---

## 4. Branch Pipeline

Fork from the main DataFrame (or from storage), process independently, then merge back.

### Join Branch
```python
pipe = TransformerPipeline(
    [
        DropColumns(columns=["c1", "c2"]),
        AddLiterals(data=[{"value": "joined", "alias": "new_col"}]),
    ],
    branch={"end": "join", "on": "idx", "how": "inner"},
)
```

### Append Branch
```python
pipe = TransformerPipeline(
    [AddLiterals(data=[{"value": "from_branch", "alias": "c1"}])],
    branch={"end": "append"},
)
```

### Dead-End Branch
```python
pipe = TransformerPipeline(
    [
        AddLiterals(data=[{"value": "audit", "alias": "audit_col"}]),
        {"store": "audit_result"},  # save to storage, not merged back
    ],
    branch={"end": "dead-end"},
)
```

### Branch from Storage
```python
pipe = TransformerPipeline(
    [SelectColumns(columns=["id", "amount"])],
    branch={"storage": "lookup_table", "end": "join", "on": "id", "how": "left"},
)
```

**Branch dict keys:**
- `end` (required): `"join"`, `"append"`, or `"dead-end"`
- `storage` (optional): key to read from nebula_storage (omit to fork from current df)
- `on` / `left_on` / `right_on`: join keys (for `end="join"`)
- `how`: join type (for `end="join"`)
- `suffix`: column suffix for join conflicts (default `"_right"`)
- `broadcast`: (Spark only) broadcast hint for join
- `skip` / `perform`: conditional execution

---

## 5. Apply-to-Rows

Filter rows matching a condition, transform only those, merge back.

```python
pipe = TransformerPipeline(
    [AddLiterals(data=[{"value": "modified", "alias": "flag"}])],
    apply_to_rows={"input_col": "amount", "operator": "gt", "value": 100},
)
```

**apply_to_rows dict keys:**
- `input_col` (required): column to evaluate
- `operator` (required): same operators as Filter **except `ne`** (not allowed — null handling is ambiguous; use `When` to create a boolean column + `operator="eq"` instead, or swap main/otherwise pipelines and use `eq`)
- `value`: comparison value
- `comparison_column`: compare with another column (mutually exclusive with `value`)
- `dead-end`: if True, matching rows are not merged back
- `skip_if_empty`: skip if no rows match
- `skip` / `perform`: conditional execution

---

## 6. Otherwise

Fallback pipeline for the unmatched rows (apply-to-rows) or the main DataFrame (branch).

```python
# With apply-to-rows: matched rows get "modified", unmatched get "original"
pipe = TransformerPipeline(
    [AddLiterals(data=[{"value": "modified", "alias": "flag"}])],
    apply_to_rows={"input_col": "idx", "operator": "gt", "value": 5},
    otherwise=AddLiterals(data=[{"value": "original", "alias": "flag"}]),
)

# With branch: branch path and main path each get different transformations
pipe = TransformerPipeline(
    [AddLiterals(data=[{"value": "branch_path", "alias": "c1"}])],
    branch={"end": "append"},
    otherwise=AddLiterals(data=[{"value": "main_path", "alias": "c1"}]),
)
```

**How `otherwise` works:**
- The `otherwise` pipeline transforms the "other" rows (apply-to-rows: unmatched rows) or the main DF (branch: the DF that wasn't forked).
- The output of `otherwise` is **promoted to the pipeline output** — it replaces the main DF, not discarded.
- When a branch/apply_to_rows is skipped (via `skip=True` or `perform=False`), the `otherwise` pipeline runs on the full input DF and its output becomes the pipeline result.
- `otherwise` accepts any step type: a single transformer instance, a list of steps, or a `TransformerPipeline`.

---

## 7. Nested Pipelines

Pipelines can contain sub-pipelines.

```python
cleaning = TransformerPipeline([
    DropNulls(columns=["id"]),
    Cast(cast={"amount": "float64"}),
], name="cleaning")

enrichment = TransformerPipeline([
    SelectColumns(columns=["id", "amount"]),
], branch={"storage": "lookup", "end": "join", "on": "id", "how": "left"})

master = TransformerPipeline([
    cleaning,
    enrichment,
    SelectColumns(columns=["id", "amount", "tier"]),
], name="master")
```

Nested lists in data are flattened:
```python
pipe = TransformerPipeline([
    transformer_a,
    [transformer_b, transformer_c],  # flattened into the same level
])
```

---

## 8. Storage Keywords

Dict keywords in the pipeline list for controlling storage.

```python
pipe = TransformerPipeline([
    Filter(input_col="status", operator="eq", value="active"),
    {"store": "active_users"},              # save current df to storage
    SelectColumns(columns=["id"]),
    {"store_debug": "debug_snapshot"},       # save only if debug mode on
    {"storage_debug_mode": True},            # toggle debug mode
    {"from_store": "active_users"},          # replace current df from storage
])
```

**String keywords (no args):**
```python
pipe = TransformerPipeline([
    "to_lazy",       # convert to lazy frame (Polars)
    Filter(...),
    "collect",       # collect lazy to eager
    "to_native",     # Narwhals -> native
    custom_fn,       # use native API
    "from_native",   # native -> Narwhals
])
```

---

## 9. Skip / Perform

Conditional pipeline execution. `skip` and `perform` are the same mechanism (inverses of each other) and can appear at different scopes: on a pipeline, on a branch/apply_to_rows dict, or on a step in YAML config. The effect is always "don't run this" — if a pipeline contains a single step, skipping the pipeline is equivalent to skipping that step.

```python
FEATURE_FLAG = False

# Skip entire pipeline
pipe = TransformerPipeline([...], skip=True)
pipe = TransformerPipeline([...], perform=False)  # equivalent

# Skip branch
pipe = TransformerPipeline(
    [...],
    branch={"end": "append", "skip": not FEATURE_FLAG},
    otherwise=[fallback_transformer],  # runs when branch is skipped
)

# Skip apply-to-rows
pipe = TransformerPipeline(
    [...],
    apply_to_rows={"input_col": "x", "operator": "gt", "value": 0, "perform": FEATURE_FLAG},
)
```

---

## 10. Interleaved (Debug Only)

Insert transformers between every step. For development/debugging only.

```python
pipe = TransformerPipeline(
    [Filter(...), SelectColumns(...), Cast(...)],
    interleaved=[AssertNotEmpty(), PrintSchema()],
    prepend_interleaved=True,   # also before first step
    append_interleaved=True,    # also after last step
)

# Or inject at runtime without modifying pipeline definition
result = pipe.run(df, after_each_step=LogDataSkew())
```

---

## 11. LazyWrapper (Runtime Parameter Resolution)

Defer transformer instantiation until execution, when storage values are available.

```python
from nebula.base import LazyWrapper
from nebula.storage import nebula_storage as ns

ns.set("threshold", 100)

pipe = TransformerPipeline([
    LazyWrapper(Filter, input_col="amount", operator="gt", value=(ns, "threshold")),
    LazyWrapper(AddLiterals, data=[{"alias": "col", "value": (ns, "dynamic_value")}]),
])
```

At runtime: `(ns, "key")` tuples are resolved to `ns.get("key")`, then the transformer is instantiated.

---

## 12. `run()` Parameters

```python
result = pipe.run(
    df,
    hooks=my_hooks,                      # optional PipelineHooks for monitoring
    resume_from="node_id",               # skip all nodes before this ID (use show(add_ids=True) to find IDs)
    show_params=True,                    # print transformer parameters during execution
    after_each_step=PrintSchema(),       # transformer or function injected after every step
)
```

- `run()` returns the same DataFrame type as the input (Pandas in → Pandas out, Polars in → Polars out, etc.).
- `resume_from` is useful for debugging failures — re-run from the step that broke.
- `after_each_step` accepts a `Transformer` instance or a callable `fn(df) -> df`. Use `functools.partial` to pass extra arguments.

---

## 13. Visualization

```python
# Text output
pipe.show()                              # basic structure
pipe.show(add_params=True)               # include transformer parameters
pipe.show(add_params=True, add_ids=True) # include node IDs (for resume_from)

# Graphviz DAG (requires graphviz + pyyaml)
dot = pipe.plot(add_params=True, add_description=True)
dot.render("pipeline", format="png")

# String representation
text = pipe.to_string(add_params=True)
```

---

## 14. Failure Recovery

On failure, the pipeline raises immediately (no retry/continue). The DataFrame before the failing step is auto-stored. This applies to all failures including assertion errors (e.g., `AssertContainsColumns`, `AssertNotEmpty`):
```python
try:
    pipe.run(df)
except Exception:
    df_before_failure = ns.get("FAIL_DF_transformer:TransformerName")
```
