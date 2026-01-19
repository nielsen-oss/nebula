"""Nebula - Declarative data transformation pipelines.

Build transformation pipelines once, run on Pandas, Polars, or PySpark.

Main exports:
    TransformerPipeline: Core class for defining pipelines
    load_pipeline: Load pipeline from YAML/JSON configuration
    nebula_storage: Runtime storage for intermediate results / debugging

Example:
    >>> from nebula import TransformerPipeline
    >>> from nebula.transformers import SelectColumns, Filter
    >>> pipeline = TransformerPipeline([
    ...     SelectColumns(columns=["id", "name"]),
    ...     Filter(input_col="id", operator="gt", value=0),
    ... ])
    >>> result = pipeline.run(df)  # Works with any backend!
"""

from nebula.pipelines.pipeline import TransformerPipeline
from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.storage import nebula_storage

__all__ = [
    "TransformerPipeline",
    "load_pipeline",
    "nebula_storage",
]
