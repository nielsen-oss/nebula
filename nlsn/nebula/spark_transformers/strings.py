"""String Manipulations: Concatenation, Regex, Formatting, etc."""

from typing import Iterable, List, Optional, Union

from pyspark.sql import functions as F

from nlsn.nebula.auxiliaries import validate_regex_pattern
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import assert_col_type

__all__ = [
    "EmptyStringToNull",
    "RegexExtract",
    "RegexReplace",
    "SplitStringToList",
    "Substring",
]


class EmptyStringToNull(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        trim: bool = False,
    ):
        """Empty strings "" are replaced with null values.

        Args:
            columns (str | list(str) | None):
                A list of the objective columns. Defaults to None.
            regex (str | None):
                Select the objective columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the objective columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            trim (bool):
                If True strips blank spaces before checking if there is an
                empty string "". The trimming is just for checking; the
                string itself in the resulting dataframe is NOT trimmed.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        self._trim: bool = trim

    def _transform(self, df):
        full_selection: List[str] = self._get_selected_columns(df)

        # Set of all the dataframe columns that are StringType
        string_columns = {c[0] for c in df.dtypes if c[1] == "string"}

        # Keep the selected columns that are StringType
        selection = [i for i in full_selection if i in string_columns]

        def _to_null(_c):
            _c_comp = F.trim(_c) if self._trim else F.col(_c)
            return F.when(_c_comp != "", F.col(_c)).otherwise(F.lit(None))

        ret = [_to_null(x).alias(x) if x in selection else x for x in df.columns]
        return df.select(*ret)


class RegexExtract(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        pattern: str,
        output_col: Optional[str] = None,
        extract: int = 0,
        drop_input_col: bool = False,
    ):
        """Extract a group using a regular expression.

        Args:
            input_col (str):
                Column to be considered during extraction.
            pattern (str):
                The regular expression to be used.
            output_col (str | None):
                Column name for extracted values. If not provided, the
                output will be stored in the 'input_col'.
            extract (int):
                The group to extract. Groups are counted by when parentheses
                are opened, left to right, starting at 1.
                Group 0 is the entire expression.
                Defaults to 0.
            drop_input_col (bool):
                Drop input column or not.
                Defaults to False.
        """
        if extract < 0:
            raise ValueError('"extract" must be > 0.')

        msg = '"drop_input_col" cannot be True if "output_col" is not provided.'
        if (not output_col) and drop_input_col:
            raise ValueError(msg)
        if (input_col == output_col) and drop_input_col:
            raise ValueError(msg)

        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col if output_col else input_col
        self._pattern: str = pattern
        self._extract: int = extract
        self._drop_input_col: bool = drop_input_col

    def _transform(self, df):
        func = F.regexp_extract(F.col(self._input_col), self._pattern, self._extract)
        df = df.withColumn(self._output_col, func)

        if self._drop_input_col and self._output_col != self._input_col:
            df = df.drop(self._input_col)

        return df


class RegexReplace(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        columns_regex: Optional[str] = None,
        columns_glob: Optional[str] = None,
        columns_startswith: Optional[Union[str, Iterable[str]]] = None,
        columns_endswith: Optional[Union[str, Iterable[str]]] = None,
        pattern: str,
        replacement: str,
    ):
        """String replacement using Regex expression.

        The input parameters
        - 'columns'
        - 'columns_regex'
        - 'columns_glob'
        - 'columns_startswith'
        - 'columns_endswith'
        are related to the column selection, to apply the regex-replace,
        whereas the parameters 'pattern' and 'replacement' are to the
        actual values regex-replace.

        Args:
            columns (str | list(str) | None):
                A list of columns to which the regex-replace will be applied.
                Defaults to None.
            columns_regex (str | None):
                Take the columns to which the regex-replace will be applied by
                using a regex pattern. Defaults to None.
            columns_glob (str | None):
                Take the columns to which the regex-replace will be applied by
                using a bash-like pattern. Defaults to None.
            columns_startswith (str | iterable(str) | None):
                Take the columns whose names start with the provided
                string(s). Defaults to None.
            columns_endswith (str | iterable(str) | None):
                Take the columns whose names end with the provided
                string(s). Defaults to None.
            pattern (str):
                The regular expression to be replaced.
            replacement (str):
                The replacement for each match.
        """
        super().__init__()
        # Keep the variable name 'columns_regex' not to get confused with the
        # *regex pattern* to match.
        self._set_columns_selections(
            columns=columns,
            regex=columns_regex,
            glob=columns_glob,
            startswith=columns_startswith,
            endswith=columns_endswith,
        )
        self._pattern: str = pattern
        self._rep: str = replacement

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)

        new_cols = []

        for c in df.columns:
            if c in selection:
                new_c = F.regexp_replace(F.col(c), self._pattern, self._rep).alias(c)
            else:
                new_c = c
            new_cols.append(new_c)

        return df.select(*new_cols)


class SplitStringToList(Transformer):
    # pylint: disable=anomalous-backslash-in-string
    def __init__(
        self,
        *,
        input_col: str,
        output_col: Optional[str] = None,
        regex: str = r"[^a-zA-Z0-9\s]",
        limit: int = -1,
        cast: str = None,
    ):  # noqa: D301
        """Splits a column of strings into a column of lists.

        The original strings are split based on a regex pattern.

        Args:
            input_col (str):
                The name of the input column containing the string values.
            output_col (str | None):
                The name of the output column to store the transformed list.
                Use input_column, if not provided. Defaults to None.
            regex (str | None):
                The regular expression pattern used to split the string values.
                Defaults to r"[^a-zA-Z0-9\s]".
            limit (int | None):
                An integer which controls the number of times the pattern is applied.
                limit > 0: The resulting array’s length will not be more than
                    limit, and the resulting array’s last entry will contain
                    all inputs beyond the last matched pattern.
                limit <= 0: pattern will be applied as many times as possible, and
                    the resulting array can be of any size.

            cast (str | None): Cast the transformed elements to a specified type.
                Defaults to array<string>.
        """
        validate_regex_pattern(regex)
        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col or input_col
        self._regex: str = regex
        self._limit: int = limit
        self._cast: str = cast

    def _transform(self, df):
        assert_col_type(df, self._input_col, "string")

        out_col = F.split(F.col(self._input_col), self._regex, self._limit)
        if self._cast:
            out_col = out_col.cast(f"array<{self._cast}>")
        return df.withColumn(self._output_col, out_col)


class Substring(Transformer):
    def __init__(
        self, *, input_col: str, output_col: str = None, start: int, length: int
    ):
        """Substring starts at 'start' and is of length 'length'.

        Args:
            input_col (str):
                The name of the input column.
            output_col (str):
                The name of the output column.
            start (int):
                The starting position of the substring.
            length (int):
                The length of the substring.
        """
        # Check if the length is within valid bounds
        if length < 1:
            raise ValueError("'Length' must be a positive integer.")

        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col or input_col
        self._start: int = start
        self._length: int = length

    def _transform(self, df):
        op = F.substring(self._input_col, self._start, self._length)
        return df.withColumn(self._output_col, op)
