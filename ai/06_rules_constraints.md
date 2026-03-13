# Nebula - Rules & Constraints for AI Code Generation

When generating Nebula pipelines, follow these rules strictly.

---

## Transformer Rules

1. **All `__init__` parameters are keyword-only.** Always use `param=value`, never positional args.
   ```python
   # CORRECT
   SelectColumns(columns=["a", "b"])

   # WRONG
   SelectColumns(["a", "b"])
   ```

2. **Transformers must be instantiated.** Pass instances, not classes.
   ```python
   # CORRECT
   TransformerPipeline([SelectColumns(columns=["a"])])

   # WRONG
   TransformerPipeline([SelectColumns])
   ```

3. **`super().__init__()` is required** in custom transformer `__init__`.

4. **Column selection patterns** — many transformers accept multiple ways to select columns. Use exactly one:
   - `columns` — explicit list
   - `regex` — regex pattern
   - `glob` — glob pattern
   - `startswith` / `endswith` — prefix/suffix match

5. **Filter operators** — the full list: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `isin`, `between`, `contains`, `startswith`, `endswith`, `like`, `rlike`, `array_contains`, `is_null`, `is_not_null`, `is_nan`, `is_not_nan`. Operators `is_null`, `is_not_null`, `is_nan`, `is_not_nan` do NOT take a `value`.

6. **Spark-only transformers** — never use in Polars/Pandas/DuckDB pipelines: `Repartition`, `CoalescePartitions`, `Persist`, `Cache`, `LogDataSkew`, `CpuInfo`, `AggregateOverWindow`, `LagOverWindow`, `ColumnsToMap`, `MapToColumns`, `SparkSqlFunction`, `SparkColumnMethod`, `SparkDropDuplicates`, `SparkExplode`.

---

## Pipeline Rules

7. **Split pipelines require `split_function`** when data is a dict with 2+ keys. The function must return a dict with the **same keys** as the data dict.

8. **`split_order` must contain exactly the same keys** as the data dict. No missing, no extra.

9. **`branch` and `apply_to_rows` are mutually exclusive** — cannot use both on the same pipeline.

10. **`otherwise` requires `branch` or `apply_to_rows`** — cannot use `otherwise` alone or with split pipelines. Also:
    - For `branch`: requires forking from primary DF (no `storage` key) and `end` != `"dead-end"`
    - For `apply_to_rows`: `dead-end` must not be True

11. **`skip` and `perform` must not contradict** — cannot set `skip=True, perform=True`.

12. **`cast_subsets_to_input_schema` and `allow_missing_columns` are mutually exclusive.**

13. **Nested lists are flattened.** `[a, [b, c]]` becomes `[a, b, c]`.

14. **Single-element dicts in data** — if data is a dict with exactly 1 key, it's treated as a keyword operation (like `{"store": "key"}`), NOT as a split pipeline.

---

## YAML Config Rules

15. **Transformer names are strings** matching class names in `nebula.transformers` or `extra_transformers`.

16. **`lazy: true` is required** when using `__ns__` markers in params. Without it, the markers are treated as plain strings.

17. **`__ns__` prefix** in YAML strings references nebula_storage: `"__ns__my_key"` becomes `(ns, "my_key")` at runtime.

18. **Loop placeholders use `<<name>>`** syntax. If the placeholder IS the entire value, type is preserved. If embedded in a string, result is string.

19. **`extra_functions` must be provided** when YAML references functions by string name (for `split_function`, `function` steps, etc.).

20. **Loop `mode`**: `"linear"` (default) requires all value lists to be same length. `"product"` creates Cartesian product.

---

## Storage Rules

21. **`ns.set(key, value)`** — by default, overwriting an existing key raises an error. Call `ns.allow_overwriting()` first if needed.

22. **Storage keys used in `Join` and `AppendDataFrame`** must be set before the pipeline reaches that step.

23. **`store_debug`** only stores if debug mode is active (`ns.allow_debug(True)`).

24. **On failure**, the pre-failure DataFrame is auto-cached with key `"FAIL_DF_transformer:TransformerName"`.

---

## Function Tuple Format

25. **Tuple format for functions in pipeline:**
    ```
    (callable,)                              # no args
    (callable, [args_list])                  # with positional args
    (callable, [args_list], {kwargs_dict})   # with args and kwargs
    (callable, [args_list], {kwargs_dict}, "description")  # with description
    ```
    The function signature must be `fn(df, *args, **kwargs) -> df`.

26. **Tuple format for transformer descriptions:**
    ```
    (transformer_instance, "description string")
    ```

---

## Dynamic Method Discovery

Do NOT hardcode lists of valid method names — they depend on the user's installed Narwhals/PySpark version. Instead, discover valid methods at runtime:

- **`WithColumns(method=...)`** — valid methods are on `nw.Expr`: `{m for m in dir(nw.col()) if not m.startswith("_") and m.islower()}`
- **`DataFrameMethod(method=...)`** — valid methods depend on eager vs lazy state:
  - Eager: `{m for m in dir(nw.DataFrame) if not m.startswith("_") and m.islower()}`
  - Lazy: `{m for m in dir(nw.LazyFrame) if not m.startswith("_") and m.islower()}`
  - Note: `LazyFrame` has a different method set than `DataFrame` (e.g., no `.head()`, has `.collect()`)
- **`SparkSqlFunction(function=...)`** — valid functions: `{m for m in dir(pyspark.sql.functions) if not m.startswith("_") and not m[0].isupper()}`
- **`SparkColumnMethod(method=...)`** — valid methods are on `pyspark.sql.Column`: `{m for m in dir(pyspark.sql.Column) if not m.startswith("_") and m.islower()}`

When unsure if a method exists, prefer well-known methods (e.g., `sort`, `unique`, `head`, `tail`, `sample`, `rename`, `explode`) or suggest the user verify with the `dir()` pattern above.

---

## Common Pitfalls

- **Forgetting `super().__init__()`** in custom transformers — breaks parameter tracking.
- **Using `ne` (not equal) in `apply_to_rows` operator** — not allowed, because null handling is ambiguous (should nulls be "not equal" or excluded?). Workaround: create a boolean column first (e.g., via `When` or `Filter`), then use `apply_to_rows` with `operator="eq"` on that boolean column. Or swap the main pipeline and `otherwise` pipeline and use `eq` instead.
- **Mixing `value` and `compare_col` in Filter** — they are mutually exclusive.
- **Setting `lazy: false` (or omitting it) with `__ns__` markers** — markers won't be resolved.
- **Using split data dict with 1 key** — interpreted as keyword, not split. Splits need 2+ keys.
- **Forgetting to pass `extra_functions`** when YAML references custom functions — causes lookup errors.
- **Using Spark transformers in non-Spark pipelines** — will raise errors at runtime.

---

## Code Generation Checklist

When generating a Nebula pipeline:

1. Identify the backend (Pandas/Polars/DuckDB/Spark) or if it should be backend-agnostic
2. Choose between code (Python) or config (YAML) format
3. For each transformation step, check if a built-in transformer exists before writing custom code
4. Use Narwhals-compatible transformers when possible for portability
5. Use `DataFrameMethod` or `WithColumns` as escape hatches before writing custom transformers
6. For YAML: ensure all custom functions/transformers are registered via `extra_functions`/`extra_transformers`
7. For splits: ensure split function returns dict with matching keys
8. For branches: choose appropriate `end` strategy (join/append/dead-end)
9. Validate: no contradictory skip/perform, no mixing branch+apply_to_rows
10. If using storage: ensure keys are set before they're read
