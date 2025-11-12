"""Time Manipulation, Conversions, and Operations."""

from datetime import datetime, timedelta
from typing import List, Optional, Set, Union

import pyspark.sql.functions as F
from pyspark.sql.types import DateType

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_is_bool,
    assert_is_string,
    assert_only_one_non_none,
    assert_valid_timezone,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import assert_col_type, get_column_data_type_name
from nlsn.nebula.ts_util import py2java_format

__all__ = [
    "DayOfWeek",
    "FromUtcTimestamp",
    "IsWeekend",
    "Timedelta",
    "TimeOfDay",
    "ToUtcTimestamp",
]



class DayOfWeek(Transformer):
    def __init__(
        self, *, input_col: str, as_string: bool, output_col: Optional[str] = None
    ):
        """Extract the day of week from a <TimestampType> column.

        The result can be string (day abbr.) or integer [1,7].
        If the input value is null, the output will be null.

        Args:
            input_col (str):
                Must be <TimestampType> column.
            as_string (bool):
                If True return a <StringType> column with values:
                    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", null
                If False return a <IntegerType> column with values:
                    1, 2, 3, 4, 5, 6, 7, null
                    Where 1 = Sunday, ... 7 = Saturday.
            output_col (str | None):
                If None, the result will replace the input_col.
        """
        assert_is_bool(as_string, "as_string")

        super().__init__()
        self._input_col: str = input_col
        self._as_string: bool = as_string
        self._output_col: Optional[str] = output_col

    def _transform(self, df):
        if self._as_string:
            dow = F.date_format(self._input_col, "E")
        else:
            dow = F.dayofweek(self._input_col)

        out_col = self._output_col if self._output_col else self._input_col
        return df.withColumn(out_col, dow)


class FromUtcTimestamp(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        output_col: Optional[str] = None,
        timezone: Optional[str] = None,
        timezone_col: Optional[str] = None,
    ):
        """Convert a UTC timestamp to a timezone-aware timestamp.

        Args:
            input_col (str):
                The input datetime to convert to timezone-aware.
                It must be <TimestampType>.
            output_col (str | None):
                The output column with the new timestamp.
                If not provided, the input column will be used as output.
                Defaults to None.
            timezone (str):
                A string detailing the time zone ID that the input should be
                adjusted to. It should be in the format of either
                region-based zone IDs or zone offsets. Region IDs must have
                the form ‘area/city’, such as ‘America/Los_Angeles’.
                Zone offsets (‘(+|-)HH:mm’) are currently not supported.
            timezone_col (str):
                Like 'timezone' arg but the timezone is now a value of the
                'timezone_col' column.

        Raises:
            ValueError: If `timezone` is unknown.
        """
        assert_only_one_non_none(timezone, timezone_col)

        if timezone:
            assert_valid_timezone(timezone)

        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col if output_col else input_col
        self._tz: Optional[str] = timezone
        self._tz_col: Optional[str] = timezone_col

    def _transform(self, df):
        assert_col_type(df, self._input_col, "timestamp")

        tz: Union[str, F.col] = self._tz if self._tz else F.col(self._tz_col)
        out = F.from_utc_timestamp(self._input_col, tz)
        return df.withColumn(self._output_col, out)



class IsWeekend(Transformer):
    def __init__(
        self,
        *,
        input_col: Optional[str] = None,
        dayofweek: Optional[str] = None,
        split_saturday_sunday: bool,
    ):
        """Add boolean column(s) indicating if a date is a working day or a weekend.

        If 'split_saturday_sunday' is True, it creates three new boolean columns:
        - 'is_working_day': True for working days (Monday to Friday), False otherwise.
        - 'is_saturday': True for Saturdays, False otherwise.
        - 'is_sunday': True for Sundays, False otherwise.

        If 'split_saturday_sunday' is False, it creates two new boolean columns:
        - 'is_working_day': True for working days (Monday to Friday), False otherwise.
        - 'is_weekend': True for weekends (Saturday and Sunday), False otherwise.

        When the 'input_col' or 'dayofweek' values are null, the new values
        will also be null.

        Args:
            input_col (str | None):
                The Input column, it must be a <TimestampType>.
            dayofweek (str | None):
                The name of the "dayofweek" column, which must be
                <IntegerType> or <LongType>.
            split_saturday_sunday (bool):
                If True, it creates separate boolean columns for Saturday and
                Sunday, namely "is_saturday" and "is_sunday".
                If False, it combines Saturday and Sunday into a single
                'is_weekend' boolean column.
        """
        assert_only_one_non_none(input_col, dayofweek)
        super().__init__()
        self._input_col: Optional[str] = input_col
        self._dayofweek: Optional[str] = dayofweek
        self._split_saturday_sunday: bool = split_saturday_sunday

    def _transform(self, df):
        if self._dayofweek:
            assert_col_type(df, self._dayofweek, {"integer", "long", "int", "bigint"})
            dow = F.col(self._dayofweek)
        else:
            dow = F.dayofweek(self._input_col)

        null_clause = F.when(dow.isNull(), F.lit(None))

        list_cond = [("is_working_day", dow.isin({2, 3, 4, 5, 6}))]

        if self._split_saturday_sunday:
            list_cond.extend([("is_saturday", dow == 7), ("is_sunday", dow == 1)])
        else:
            list_cond.append(("is_weekend", dow.isin({1, 7})))

        new_col_clauses: List[F.col] = []
        new_col_names: Set[str] = set()
        for col_name, cond in list_cond:
            new_col_names.add(col_name)
            day_clause = F.when(cond, F.lit(True)).otherwise(F.lit(False))
            clause = null_clause.otherwise(day_clause)
            new_col_clauses.append(clause.alias(col_name))

        orig_cols = [i for i in df.columns if i not in new_col_names]
        return df.select(*orig_cols, *new_col_clauses)


class Timedelta(Transformer):
    def __init__(
        self,
        *,
        output_col: str,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        input_col: Optional[str] = None,
        date: Optional[Union[str, datetime]] = None,
        input_dt_format: Optional[str] = None,
        output_dt_format: Optional[str] = None,
    ):
        """Add a timedelta to a certain datetime.

        Args:
            output_col (str):
                Name of the new column to create.
            days (int):
                The number of days to add/subtract.
                Defaults to 0.
            hours (int):
                The number of hours to add/subtract.
                Defaults to 0.
            minutes (int):
                The number of minutes to add/subtract.
                Defaults to 0.
            seconds (int):
                The number of seconds to add/subtract.
                Defaults to 0.
            input_col (str | None):
                The name of the input column to add/subtract 'days' from.
                If None, 'date' argument is used.
            date (str | datetime.datetime | datetime.date | None):
                The starting date to add/subtract 'days' from.
                If None, the 'input_column' argument is used.
            input_dt_format (str | None):
                Specify the input datetime format for conversion, which is
                mandatory when the input column type is <StringType>.
                If the input date is the argument 'date', it must be in python
                format like "%Y-%m-%d %H:%M:%S".
                Otherwise, if the input date is a column, it can be either a
                python or a java format specifier.
            output_dt_format (str | None):
                Specify the output datetime format for conversion if requested.
                If the input date is the argument 'date', it must be in python
                format like "%Y-%m-%d %H:%M:%S".
                Otherwise, if the input date is a column, it can be either a
                python or a java format specifier.
        """
        assert_only_one_non_none(input_col, date)

        if isinstance(date, str):
            date = datetime.strptime(date, input_dt_format)

        super().__init__()
        self._days: int = days
        self._hours: int = hours
        self._minutes: int = minutes
        self._seconds: int = seconds
        self._output_col: str = output_col
        self._input_col: Optional[str] = input_col
        self._date: Optional[datetime] = date
        self._input_dt_format: Optional[str] = input_dt_format
        self._output_dt_format: Optional[str] = output_dt_format

    def _from_literal(self, df):
        output: Union[str, datetime]
        td: timedelta = timedelta(
            days=self._days,
            hours=self._hours,
            minutes=self._minutes,
            seconds=self._seconds,
        )
        output = self._date + td
        if self._output_dt_format:
            output = output.strftime(self._output_dt_format)
        return df.withColumn(self._output_col, F.lit(output))

    def _transform(self, df):
        if self._date:
            return self._from_literal(df)

        input_type: str = get_column_data_type_name(df, self._input_col)

        # If the input column is a <StringType> cast to <TimestampType>.
        ts: F.col
        if input_type == "string":
            if self._input_dt_format is None:
                msg = '"{}" must be provided if the input column is <{}>'
                raise AssertionError(msg.format("input_dt_format", "string"))
            # Convert the format to a java one if necessary.
            fmt = py2java_format(self._input_dt_format)
            ts = F.to_timestamp(self._input_col, format=fmt)
        elif input_type == "timestamp":
            ts = F.col(self._input_col)
        else:
            raise TypeError("Input column must be <string> or <timestamp>")

        expr: str = " ".join(
            [
                "INTERVAL",
                f"{self._days} DAYS",
                f"{self._hours} HOURS",
                f"{self._minutes} MINUTES",
                f"{self._seconds} SECONDS",
            ]
        )

        output: F.col = ts + F.expr(expr)
        if self._output_dt_format:
            # Convert the format to a java one if necessary.
            fmt = py2java_format(self._output_dt_format)
            output = F.date_format(output, fmt)
        return df.withColumn(self._output_col, output)


class TimeOfDay(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        output_col: Optional[str] = None,
        unit: str,
        from_start: bool = False,
    ):
        """Extract the specified time unit (hour, minute, or second) from a datetime.

        The output is an integer of the specified time unit, and the input
        column can be either a 'string' or a 'datetime' type.
        It provides the flexibility to extract the time unit either as is or
        from the beginning of the day, depending on the 'from_start' parameter.

        Args:
            input_col (str):
                The name of the input timestamp or string column.
            output_col (str):
                The name of the output column. If not provided, the output
                value replaces the input. Defaults to None.
            unit (str):
                The time unit to extract ('hour', 'minute', or 'second').
            from_start (bool):
                If True, extract the hour/minute/second from the start of
                the day. Defaults to False.
        """
        assert_allowed(unit, {"hour", "minute", "second"}, "unit")

        super().__init__()
        self._unit: str = unit
        self._input_col: str = input_col
        self._output_col: str = output_col or input_col
        self._from_start: bool = from_start

    def _transform(self, df):
        assert_col_type(df, self._input_col, {"string", "timestamp"})

        hour = F.hour(self._input_col)
        minute = F.minute(self._input_col)
        second = F.second(self._input_col)

        if self._from_start:
            time_func = {
                "hour": hour,
                "minute": (hour * 60) + minute,
                "second": (hour * 3600) + (minute * 60) + second,
            }
        else:
            time_func = {
                "hour": hour,
                "minute": minute,
                "second": second,
            }

        return df.withColumn(self._output_col, time_func[self._unit])



class ToUtcTimestamp(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        output_col: Optional[str] = None,
        timezone: Optional[str] = None,
        timezone_col: Optional[str] = None,
    ):
        """Convert a timezone-aware timestamp to UTC.

        Args:
            input_col (str):
                The input datetime that is timezone-aware.
                It must be <TimestampType>.
            output_col (str | None):
                The output column with the new timestamp in UTC.
                If not provided, the input column will be used as output.
                Defaults to None.
            timezone (str):
                A string detailing the time zone ID that the input should be
                converted from. It should be in the format of either
                region-based zone IDs or zone offsets. Region IDs must have
                the form ‘area/city’, such as ‘America/Los_Angeles’.
                Zone offsets (‘(+|-)HH:mm’) are currently not supported.
            timezone_col (str):
                Like 'timezone' arg, but the timezone is now a value of the
                'timezone_col' column.

        Raises:
            ValueError: If `timezone` is unknown.
        """
        assert_only_one_non_none(timezone, timezone_col)

        if timezone:
            assert_valid_timezone(timezone)

        super().__init__()
        self._input_col: str = input_col
        self._output_col: str = output_col if output_col else input_col
        self._tz: Optional[str] = timezone
        self._tz_col: Optional[str] = timezone_col

    def _transform(self, df):
        assert_col_type(df, self._input_col, "timestamp")
        tz: Union[str, F.col] = self._tz if self._tz else F.col(self._tz_col)
        out = F.to_utc_timestamp(self._input_col, tz)
        return df.withColumn(self._output_col, out)
