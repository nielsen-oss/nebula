"""Spark Transformers for handling TemporaryViews and Nebula Storage."""

from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import get_spark_session, table_is_registered
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "DropTable",
    "GetTable",
    "RegisterTable",
    "SparkTableToStorage",
]


def _check_is_registered(table: str, nebula_cache: bool, df):
    msg = f"{table} is not registered."

    if nebula_cache:
        if not ns.isin(table):
            raise AssertionError(msg)
        return

    spark_session = get_spark_session(df)

    if not table_is_registered(table, spark_session):
        raise AssertionError(msg)


class DropTable(Transformer):
    def __init__(
            self, *, table: str, ignore_error: bool = True, nebula_cache: bool = True
    ):
        """Drop a stored dataframe and return the input dataframe untouched.

        Args:
            table (str):
                Table name.
            ignore_error (bool):
                If True don't raise AssertionError if the table does not exist.
                Defaults to True
            nebula_cache (bool):
                If True, drop the dataframe from nebula cache, otherwise
                from spark temporary views. Defaults to True.
        """
        super().__init__()
        self._table: str = table
        self._ignore_error: bool = ignore_error
        self._nebula_cache: bool = nebula_cache

    def _transform_spark(self, df):
        if not self._ignore_error:
            _check_is_registered(self._table, self._nebula_cache, df)

        if self._nebula_cache:
            ns.clear(self._table)
            return df

        spark_session = get_spark_session(df)
        spark_session.catalog.dropTempView(self._table)
        return df


class GetTable(Transformer):
    def __init__(self, *, table: str, nebula_cache: bool = True):
        """Retrieve a stored dataframe and return it.

        The input dataframe remains untouched.

        Args:
            table (str):
                Table name.
            nebula_cache (bool):
                If True, retrieve the dataframe from nebula cache, otherwise
                from spark temporary views. Defaults to True.
        """
        super().__init__()
        self._table: str = table
        self._nebula_cache: bool = nebula_cache

    def _transform_spark(self, df):
        _check_is_registered(self._table, self._nebula_cache, df)

        if self._nebula_cache:
            return ns.get(self._table)

        spark_session = get_spark_session(df)
        return spark_session.table(self._table)


class RegisterTable(Transformer):
    def __init__(self, *, table: str, mode: str = "error", nebula_cache: bool = True):
        """Register a dataframe and return the input one untouched.

        Args:
            table (str):
                Table name.
            mode (str):
                Specifies the behavior when data or table already exists.
                - overwrite: Overwrite existing data.
                - error: Throw an exception if data already exists.
                Defaults to "error".
            nebula_cache (bool):
                If True register the dataframe to nebula cache, otherwise
                to spark temporary views. Defaults to True.
        """
        super().__init__()
        self._table: str = table
        self._mode: str = mode
        self._nebula_cache: bool = nebula_cache

    def _transform_spark(self, df):
        msg_err = f"{self._table} is already registered"

        if self._nebula_cache:
            if self._mode == "error":
                if ns.isin(self._table):
                    raise AssertionError(msg_err)
            ns.set(self._table, df)
            return df

        spark_session = get_spark_session(df)

        if self._mode == "error":
            if table_is_registered(self._table, spark_session):
                raise AssertionError(msg_err)

        df.createOrReplaceTempView(self._table)
        return df


class SparkTableToStorage(Transformer):
    def __init__(self, *, input_table: str, store_key: str):
        """Read a Spark registered table and store it into nebula storage.

        Return the input dataframe untouched.

        Args:
            input_table (str):
                Spark registered table name.
            store_key (str):
                Name for the table in Nebula storage.
        """
        super().__init__()
        self._input_table: str = input_table
        self._store_key: str = store_key

    def _transform_spark(self, df):
        spark_session = get_spark_session(df)
        df_registered = spark_session.table(tableName=self._input_table)
        ns.set(self._store_key, df_registered)
        return df
