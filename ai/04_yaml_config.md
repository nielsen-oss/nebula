# Nebula - YAML/Config Format

Pipelines can be defined as Python dicts (from YAML/JSON) and loaded via `load_pipeline()`.

```python
from nebula import load_pipeline

pipe = load_pipeline(
    config,                                   # dict or path to YAML
    extra_functions=[split_fn, custom_fn],    # custom functions by name
    extra_transformers=[my_module],           # modules or dicts with custom transformers
    evaluate_loops=True,                       # enable loop expansion
)
result = pipe.run(df)
```

---

## Basic Structure

```yaml
name: "pipeline-name"          # optional
pipeline:                       # required - list of steps
  - transformer: SelectColumns
    params:
      columns: [id, amount, status]

  - transformer: Filter
    params:
      input_col: status
      operator: eq
      value: active

  - transformer: Cast
    params:
      cast:
        amount: float64
```

---

## Step Types

### Transformer Step
```yaml
- transformer: TransformerClassName    # looked up in nebula.transformers + extra_transformers
  params:                               # keyword arguments for __init__
    param1: value1
    param2: value2
  description: "Optional description"
  lazy: false                           # true = defer instantiation (LazyWrapper)
  skip: false                           # true = skip this step
  perform: true                         # false = skip this step
```

### Function Step
```yaml
- function: function_name              # looked up in extra_functions
  args: [arg1, arg2]                    # positional args (after df)
  kwargs:                               # keyword args
    key1: value1
  description: "Optional"
  skip: false
```

### Storage Keywords
```yaml
- store: storage_key                    # save current df
- store_debug: debug_key               # save only if debug mode on
- from_store: storage_key              # load df from storage (replaces current)
- storage_debug_mode: true             # toggle debug mode
```

### String Keywords
```yaml
- "to_native"                          # Narwhals -> native
- "from_native"                        # native -> Narwhals
- "collect"                            # collect lazy frames
- "to_lazy"                            # convert to lazy frames
```

---

## Nested / Split Pipeline in Config

```yaml
name: "outer-pipeline"
pipeline:
  # Step 1: a flat sub-pipeline
  - pipeline:
      - transformer: DataFrameMethod
        params:
          method: unique
      - transformer: SelectColumns
        params:
          columns: [a, b]
    name: "cleaning"

  # Step 2: a split sub-pipeline
  - pipeline:
      high:
        - transformer: AddLiterals
          params:
            data:
              - {alias: tier, value: premium}
      low:
        - transformer: AddLiterals
          params:
            data:
              - {alias: tier, value: standard}
    split_function: my_split_function     # name looked up in extra_functions
    name: "split-by-tier"
    cast_subsets_to_input_schema: true
```

---

## Branch in Config

```yaml
pipeline:
  - transformer: DropColumns
    params:
      columns: [temp_col]
  - transformer: AddLiterals
    params:
      data:
        - {alias: new_col, value: joined}
branch:
  end: join
  on: idx
  how: inner
  # Optional:
  # storage: source_key
  # skip: false
  # perform: true
```

---

## Apply-to-Rows in Config

```yaml
pipeline:
  - transformer: AddLiterals
    params:
      data:
        - {alias: flag, value: modified}
apply_to_rows:
  input_col: amount
  operator: gt
  value: 100
otherwise:
  - transformer: AddLiterals
    params:
      data:
        - {alias: flag, value: original}
```

---

## Lazy Parameters (`__ns__` Markers)

In YAML, reference nebula_storage values with the `__ns__` prefix:

```yaml
- transformer: AddLiterals
  lazy: true                            # MUST be true for lazy resolution
  params:
    data:
      - alias: dynamic_col
        value: "__ns__my_storage_key"   # resolved at runtime to ns.get("my_storage_key")
```

The `__ns__` prefix works at any nesting depth. The `lazy: true` flag is required.

---

## Loops (Dynamic Expansion)

Generate repeated pipeline steps with parameter substitution.

### Linear Mode
Values lists must have the same length. Creates N iterations.

```yaml
- loop:
    mode: linear                        # default
    values:
      names: [flag_a, flag_b, flag_c]
      codes: [1, 2, 3]
    transformer: AddLiterals
    params:
      data:
        - alias: "<<names>>"           # substituted per iteration
          value: "<<codes>>"           # substituted per iteration
```

Produces 3 `AddLiterals` steps: `(flag_a, 1)`, `(flag_b, 2)`, `(flag_c, 3)`.

### Product Mode
Cartesian product of all value combinations.

```yaml
- loop:
    mode: product
    values:
      x: [1, 2]
      y: [a, b]
    transformer: AddLiterals
    params:
      data:
        - alias: "col_<<x>>_<<y>>"
          value: "<<x>>"
```

Produces 4 steps: `(1,a)`, `(1,b)`, `(2,a)`, `(2,b)`.

### Loop with Sub-Pipeline

```yaml
- loop:
    mode: linear
    values:
      algos: [algo_X, algo_Y]
    branch:
      end: join
      how: inner
      on: join_col
    pipeline:                            # full sub-pipeline per iteration
      - transformer: SelectColumns
        params:
          columns: join_col
      - transformer: AddLiterals
        params:
          data:
            - alias: "<<algos>>"
              value: "<<algos>>"
```

### Nested Loops

Inner loop values override outer loop values with the same key.

```yaml
- loop:
    values:
      x: [1, 2]
    pipeline:
      - transformer: TR_A
        params: {col: "<<x>>"}
      - loop:
          values:
            x: [10, 20]              # overrides outer x
            y: [30, 40]
          transformer: TR_B
          params: {a: "<<x>>", b: "<<y>>"}
```

### Placeholder Rules
- `"<<param>>"` as entire value: **type preserved** (int stays int)
- `"prefix_<<param>>"` embedded in string: **converted to string**

---

## Full Pipeline Config Options

All `TransformerPipeline.__init__` parameters are valid at the config level:

```yaml
name: "pipeline-name"
pipeline: [...]
df_input_name: "Input DF"
df_output_name: "Output DF"

# Split options
split_function: func_name
split_order: [split_a, split_b]
split_apply_after_splitting:
  - transformer: AssertNotEmpty
split_apply_before_appending:
  - transformer: DataFrameMethod
    params:
      method: unique
splits_no_merge: [audit_split]
splits_skip_if_empty: [rare_case]
cast_subsets_to_input_schema: false
allow_missing_columns: false
repartition_output_to_original: false
coalesce_output_to_original: false

# Branch options
branch: {end: join, on: id, how: inner}
apply_to_rows: {input_col: x, operator: gt, value: 0}
otherwise: [...]

# Interleaved
interleaved:
  - transformer: AssertNotEmpty
prepend_interleaved: false
append_interleaved: false

# Conditional
skip: false
perform: true
```

---

## Transformer Name Resolution (YAML Only)

This resolution only applies when loading from YAML/config. In the Python API, you pass transformer instances directly — no registration or name lookup is needed.

1. Built-in transformers from `nebula.transformers` (e.g., `SelectColumns`, `Filter`, `Cast`)
2. `extra_transformers` — modules or `{"Name": Class}` dicts passed to `load_pipeline()`, higher priority
3. String names in `transformer:` field are matched against these registries

---

## Note on Templating

Nebula does not include a templating engine. Teams that need dynamic config generation (e.g., multi-environment or multi-country) typically use Jinja2 or similar tools to render YAML before passing it to `load_pipeline()`. This is entirely outside Nebula.
