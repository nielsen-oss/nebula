## Standard And Conventions

All the transformers *signatures*, *docstrings* and *conventions* are **unit-tested** before being pushed into the main branch.

- All transformers are **subclasses** of `pyspark.ml.Transformer`
- **Positional arguments** are not allowed in the `__init__` constructor
- The method `transform` accepts only **one positional argument** and nothing else
