"""Module to test 'ts_util' functions."""

import pytest

from nlsn.nebula.ts_util import py2java_format


def test_py2java_format_microseconds():
    """Tests py2java_format with a microsecond format specifier."""
    assert py2java_format("%f") == "SSSSSS"


def test_py2java_format_am_pm():
    """Tests py2java_format with an AM/PM format specifier."""
    assert py2java_format("%p") == "a"


def test_py2java_format_weekday():
    """Tests py2java_format with a weekday format specifier."""
    assert py2java_format("%A") == "EEEE"


def test_py2java_format_short_weekday():
    """Tests py2java_format with a short weekday format specifier."""
    assert py2java_format("%a") == "EEE"


def test_py2java_format_timezone():
    """Tests py2java_format with a timezone format specifier."""
    assert py2java_format("%Z") == "z"


def test_py2java_format_mixed():
    """Tests py2java_format with a mix of Python and Java format specifiers."""
    assert py2java_format("%Y-MM-dd %H:%M:%S") == "yyyy-MM-dd HH:mm:ss"


@pytest.mark.parametrize(
    "input_format, expected_output",
    [
        ("%Y-%m-%d", "yyyy-MM-dd"),
        ("%H:%M:%S", "HH:mm:ss"),
        ("%Y-%m-%d %H:%M:%S", "yyyy-MM-dd HH:mm:ss"),
        ("yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm:ss"),
        ("%Y-%m-%d %H:%M:%S.%f", "yyyy-MM-dd HH:mm:ss.SSSSSS"),
    ],
)
def test_py2java_format_parametrized(input_format, expected_output):
    """Parametrized test for py2java_format with various input formats."""
    assert py2java_format(input_format) == expected_output


def test_py2java_format_invalid_specifier():
    """Tests py2java_format with an invalid format specifier."""
    with pytest.raises(KeyError):
        py2java_format("%Q")


def test_py2java_format_year():
    """Tests py2java_format with a year format specifier."""
    assert py2java_format("%Y") == "yyyy"


def test_py2java_format_month():
    """Tests py2java_format with a month format specifier."""
    assert py2java_format("%m") == "MM"


def test_py2java_format_day():
    """Tests py2java_format with a day format specifier."""
    assert py2java_format("%d") == "dd"


def test_py2java_format_hours():
    """Tests py2java_format with an hour format specifier."""
    assert py2java_format("%H") == "HH"


def test_py2java_format_minutes():
    """Tests py2java_format with a minute format specifier."""
    assert py2java_format("%M") == "mm"


def test_py2java_format_seconds():
    """Tests py2java_format with a second format specifier."""
    assert py2java_format("%S") == "ss"


def test_py2java_date():
    """Tests py2java_format with a date format specifier."""
    assert py2java_format("%Y-%m-%d") == "yyyy-MM-dd"


def test_py2java_time():
    """Tests py2java_format with a time format specifier."""
    assert py2java_format("%H:%M:%S") == "HH:mm:ss"


def test_py2java_datetime():
    """Tests py2java_format with a combination of date and time format specifiers."""
    assert py2java_format("%Y-%m-%d %H:%M:%S") == "yyyy-MM-dd HH:mm:ss"


def test_py2java_already_java():
    """Tests py2java_format given a java format."""
    assert py2java_format("yyyy-MM-dd HH:mm:ss") == "yyyy-MM-dd HH:mm:ss"
