"""Dataframe Schema Operations: Casting, Nullability, etc.

These transformers are not supposed to act directly on values, but
some operations (i.e., cast) can affect them.
"""

from typing import Dict, Iterable, List, Optional, Union

import pyspark.sql.functions as F
from pyspark.sql.types import StructField, StructType

from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import get_spark_session

__all__ = [
    "Cast",
    "ChangeFieldsNullability",
]


class Cast(Transformer):
    def __init__(self, *, cast: Dict[str, str]):
        """Cast fields to the requested data-types.

        Args:
            cast (dict(str, str)):
                Cast dictionary <column -> type>.
        """
        super().__init__()
        self._cast: Dict[str, str] = cast

    def _transform(self, df):
        cols = [
            F.col(c).cast(self._cast[c]).alias(c) if c in self._cast else c
            for c in df.columns
        ]
        return df.select(*cols)


class ChangeFieldsNullability(Transformer):
    def __init__(
        self,
        *,
        nullable: bool,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        assert_non_nullable: bool = False,
        persist: bool = False,
    ):
        """Change field(s) nullability.

        It will convert the input dataframe into rdd, then convert back to
        a dataframe with the new schema, therefore con be a slow operation.

        Args:
            nullable (bool):
                Nullability.
            columns (str | list(str) | None):
                A list of columns to convert to nullable. Defaults to None.
            regex (str | None):
                Take the columns to convert to nullable by using a regex
                pattern. Defaults to None.
            glob (str | None):
                Take the columns to convert to nullable by using a bash-like
                pattern. Defaults to None.
            startswith (str | iterable(str) | None):
                Take the columns to convert whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Take the columns to convert whose names end with the provided
                string(s). Defaults to None.
            assert_non_nullable (bool):
                If the user changes one or more fields from nullable to
                non-nullable but the columns actually contain null value,
                spark will raise an exception solely during the first eager
                operation, not the dataframe creation phase.
                Setting this parameter to True, the transformer triggers an
                eager evaluation (.count()), raising the possible errors
                at this stage.
                This parameter is ignored if 'nullable' = True as all the
                columns can be nullable.
                Defaults to False.
            persist (bool):
                This parameter is considered only if:
                'nullable' = False
                'assert_non_nullable' = True
                It persists the dataframe if the eager evaluation is triggered.
                Defaults to False.
        """
        super().__init__()
        self._nullable: bool = nullable
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        self._assert_non_nullable: bool = assert_non_nullable
        self._persist: bool = persist

    def _transform(self, df):
        to_change: List[str] = self._get_selected_columns(df)

        field: StructField
        new_fields: List[StructField] = []

        for field in df.schema:
            name: str = field.name
            if name in to_change:
                new_field = StructField(name, field.dataType, self._nullable)
                new_fields.append(new_field)
            else:
                new_fields.append(field)

        ss = get_spark_session(df)
        df_ret = ss.createDataFrame(df.rdd, StructType(new_fields))

        if (not self._nullable) and self._assert_non_nullable:
            # Check whether any null exists in the new non-null fields.
            if self._persist:
                df_ret = df_ret.cache()
            df_ret.select(to_change).count()

        return df_ret
