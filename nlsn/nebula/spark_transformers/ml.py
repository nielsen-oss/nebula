"""Transformers for Machine Learning Features."""

from typing import List, Optional

# Prepend the underscore in Bucketizer otherwise the unit-test
# checks whether it is declared in __all__.
from pyspark.ml.feature import Bucketizer as _Bucketizer

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer

__all__ = [
    "ColumnBucketizer",
]


class ColumnBucketizer(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        buckets: list,
        add_infinity_buckets: bool,
        output_col: Optional[str] = None,
        handle_invalid: str = "error",
    ):
        """Maps a column of continuous features to a column of feature buckets.

        Bucketizes the values in the specified input column and writes the
        result to the designated output column using the provided bucket
        thresholds.

        Args:
            input_col (str):
                The name of the input column containing the values to
                be bucketized.
            output_col (str | None):
                The name of the output column where the bucketized values
                will be stored.
                If not provided, the results will be stored in the input column.
            buckets (list(int | float)):
                A list of numeric values representing the bucket thresholds.
            add_infinity_buckets (bool):
                 If True, add negative and positive infinity buckets to the
                 thresholds.
            handle_invalid (str | None):
                How to handle invalid input values.
                Must be one of "keep" or "error".
                Defaults to "error."

        Raises:
        - ValueError: If the 'handle_invalid' parameter is not 'keep' or 'error'.
        - ValueError: If the 'buckets' parameter is an empty list.
        - Py4JJavaError: If values in 'input_col' are outside the buckets and
            'add_infinity_buckets' is set to False.
        """
        assert_allowed(handle_invalid, {"keep", "error"}, "handle_invalid")

        if not buckets:
            raise ValueError('"buckets" cannot be empty.')

        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col if output_col else input_col
        # Convert the buckets to numerics
        self._buckets: List[float] = [*map(float, buckets)]
        if add_infinity_buckets:
            self._buckets.insert(0, float("-inf"))
            self._buckets.append(float("inf"))

        self._handle_invalid: str = handle_invalid

    def _transform(self, df):
        # Define bucketizer
        bucketizer = _Bucketizer(
            splits=self._buckets,
            inputCol=self._input_col,
            outputCol="_output_bucket_col_",
            handleInvalid=self._handle_invalid,
        )

        ret = bucketizer.transform(df)
        return ret.withColumnRenamed("_output_bucket_col_", self._output_col)
