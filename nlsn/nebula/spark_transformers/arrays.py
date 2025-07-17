"""Array Columns Manipulations."""

from typing import List, Optional, Union

from pyspark.sql import functions as F

from nlsn.nebula.base import Transformer

__all__ = [
    "EmptyArrayToNull",
    "Sequence",
]


class EmptyArrayToNull(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
    ):
        """Empty arrays (0 len) are replaced with null.

        This operation is ignored on non-ArrayType columns.

        Args:
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
        """
        super().__init__()
        self._set_columns_selections(columns=columns, regex=regex, glob=glob)

    @staticmethod
    def _to_null(_x) -> F.col:
        """Return None if the size of the array/map is 0."""
        return F.when(F.size(F.col(_x)) > 0, F.col(_x)).otherwise(F.lit(None))

    def _transform(self, df):
        selection: set = set(self._get_selected_columns(df))
        convert = set()
        for field in df.schema:
            if field.name not in selection:
                continue
            if field.dataType.typeName() == "array":
                convert.add(field.name)

        cols = [self._to_null(i).alias(i) if i in convert else i for i in df.columns]

        return df.select(*cols)


class Sequence(Transformer):
    def __init__(
        self,
        *,
        column: str,
        start: Union[int, str],
        stop: Union[int, str],
        step: Union[int, str] = 1,
    ):
        """Generate a sequence of integers from start to stop.

        Args:
            column (str):
                Name of the new column containing the sequence.
            start (int | str):
                Start of sequence.
            stop (int | str):
                End of sequence.
            step (int | str):
                Increment step of the sequence. Defaults to 1.
        """
        self._assert_args(start, stop, step)
        super().__init__()
        self._column: str = column
        self._start: Union[int, str] = start
        self._stop: Union[int, str] = stop
        self._step: Union[int, str] = step

    @staticmethod
    def _assert_args(start, stop, step):
        if not all(isinstance(i, (int, str)) for i in [start, stop]):
            msg = "'start', and 'stop' must be <integers> or <strings>."
            raise TypeError(msg)

        if step is None:
            return

        if not isinstance(step, (int, str)):
            msg = "If 'step' is provided, it must be <integer> or <string>."
            raise TypeError(msg)

    def _transform(self, df):
        def _make(_x):
            if isinstance(_x, str):
                return _x
            if _x is None:
                return None  # Don't return F.lit(None)
            return F.lit(_x)

        seq = F.sequence(_make(self._start), _make(self._stop), _make(self._step))
        return df.withColumn(self._column, seq)
