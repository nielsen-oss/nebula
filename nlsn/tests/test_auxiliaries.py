"""Test auxiliaries."""

import itertools
import random
from string import ascii_lowercase
from typing import List, Optional

import pytest

from nlsn.nebula.auxiliaries import *


class TestAssertAllowed:
    """Test suite for the 'assert_allowed' function."""

    @pytest.mark.parametrize("value", ["a", 1])
    def test_valid(self, value):
        """Tests with valid values."""
        assert_allowed(value, {"a", 1, 4, False}, "name")

    @pytest.mark.parametrize("value", ["3.1", "1", 3])
    def test_invalid(self, value):
        """Tests with invalid values."""
        with pytest.raises(ValueError):
            assert_allowed(value, {"a", 1, 4, False}, "name")


def test_assert_at_least_one_non_null():
    """Test 'assert_at_least_one_non_null' function."""
    a, b, c = None, None, None
    with pytest.raises(AssertionError):
        assert_at_least_one_non_null(a, b, c)

    d, e = None, True
    assert_at_least_one_non_null(d, e)

    d, e = None, False
    assert_at_least_one_non_null(d, e)

    i = False
    assert_at_least_one_non_null(i)


def test_assert_at_most_one_args():
    """Test 'assert_at_most_one_true' function."""
    a, b, c = None, None, None
    assert_at_most_one_args(a, b, c)

    d, e = None, True
    assert_at_most_one_args(d, e)

    f, g, h = 1, None, True
    with pytest.raises(AssertionError):
        assert_at_most_one_args(f, g, h)

    i, j, k = False, False, False
    assert_at_most_one_args(i, j, k)

    l, m = "letter", 1
    with pytest.raises(AssertionError):
        assert_at_most_one_args(l, m)


class TestAssertCmp:
    @pytest.mark.parametrize(
        "value, op, value_cmp, name",
        [
            (10, "eq", 10, "ten"),
            (5, "ne", 10, "five"),
            (15, "ge", 10, "fifteen"),
            (10, "ge", 10, "ten_ge"),
            (20, "gt", 10, "twenty"),
            (5, "le", 10, "five_le"),
            (10, "le", 10, "ten_le"),
            (9, "lt", 10, "nine"),
        ],
    )
    def test_assert_cmp_valid(self, value, op, value_cmp, name):
        """Valid cases."""
        assert_cmp(value, op, value_cmp, name)

    @pytest.mark.parametrize(
        "value, op, value_cmp, name",
        [
            (10, "eq", 11, "ten"),
            (10, "ne", 10, "ten"),
            (5, "ge", 10, "five"),
            (10, "gt", 10, "ten"),
            (15, "le", 10, "fifteen"),
            (10, "lt", 10, "ten"),
        ],
    )
    def test_assert_cmp_value_error(self, value, op, value_cmp, name):
        """Invalid cases."""
        with pytest.raises(ValueError):
            assert_cmp(value, op, value_cmp, name)


def test_assert_only_one_not_none():
    """Test 'assert_only_one_non_none' function."""
    a, b, c = None, None, None
    with pytest.raises(AssertionError):
        assert_only_one_non_none(a, b, c)

    d, e = None, 0
    assert_only_one_non_none(d, e)

    f, g, h = 0, None, 0
    with pytest.raises(AssertionError):
        assert_only_one_non_none(f, g, h)

    i, j = "letter", 1
    with pytest.raises(AssertionError):
        assert_only_one_non_none(i, j)

    k, l = "letter", None
    assert_only_one_non_none(k, l)


class TestAssertIsBool:
    """Test suite for the 'assert_is_bool' function."""

    @pytest.mark.parametrize("value", [True, False])
    def test_valid(self, value):
        """Tests for valid boolean values."""
        assert_is_bool(value, "test_param")

    @pytest.mark.parametrize("value", ["3.1", 1, 0])
    def test_non_bool(self, value):
        """Tests for non integer values."""
        with pytest.raises(TypeError):
            assert_is_bool(value, "test_param")


class TestAssertIsInteger:
    """Test suite for the 'assert_is_integer' function."""

    @pytest.mark.parametrize("value", [10, 0, -5, -3.0])
    def test_valid(self, value):
        """Tests for valid integer-like values."""
        assert_is_integer(value, "test_param")

    @pytest.mark.parametrize("value", ["hello", None, [1, 2], {"a": 1}])
    def test_non_numeric(self, value):
        """Tests for non numeric values."""
        with pytest.raises(TypeError):
            assert_is_integer(value, "test_param")

    @pytest.mark.parametrize("value", ["3", 1.1, -0.1])
    def test_non_integer(self, value):
        """Tests for non integer values."""
        with pytest.raises(TypeError):
            assert_is_integer(value, "test_param")


class TestAssertIsNumeric:
    """Test suite for the 'assert_is_numeric' function."""

    @pytest.mark.parametrize("value", [10, 0, -3.0])
    def test_valid(self, value):
        """Tests for valid numeric values."""
        assert_is_numeric(value, "test_param")

    @pytest.mark.parametrize("value", ["hello", "3.1", None, [1, 2], {"a": 1}])
    def test_non_numeric(self, value):
        """Tests for non numeric values."""
        with pytest.raises(TypeError):
            assert_is_numeric(value, "test_param")


class TestAssertIsString:
    """Test suite for the 'assert_is_string' function."""

    def test_valid(self):
        """Tests for valid strings."""
        assert_is_string("x", "test_param")

    @pytest.mark.parametrize("value", [1, None, [1, 2], {"a": 1}])
    def test_non_strings(self, value):
        """Tests for non-string values."""
        with pytest.raises(TypeError):
            assert_is_string(value, "test_param")


def test_assert_valid_timezone():
    """Test 'assert_valid_timezone' function."""
    assert_valid_timezone("Europe/Rome")
    with pytest.raises(ValueError):
        assert_valid_timezone("wrong")


@pytest.mark.parametrize(
    "req, cols", [[["c3"], ["c1", "c2"]], [["c1", "c3"], ["c1", "c2"]]]
)
def test_check_if_columns_are_present(req, cols):
    """Test 'check_if_columns_are_present' function."""
    with pytest.raises(AssertionError):
        check_if_columns_are_present(req, cols)


@pytest.mark.parametrize("names", [None, ["li_1", "li_2", "li_3"], ["li_1"]])
def test_compare_lists_of_string(names):
    """Test 'compare_lists_of_string' function."""
    li_1 = ["hello", "a", "long_string"]
    li_2 = ["long_string", "bb"]
    li_3 = ["a", "long_string"]

    if names:
        exp = [
            "li_1        | li_2        | li_3       ",
            "---------------------------------------",
        ]
    else:
        exp = []

    exp.extend(
        [
            "a           | ########### | a          ",
            "########### | bb          | ###########",
            "hello       | ########### | ###########",
            "long_string | long_string | long_string",
        ]
    )

    if names is not None and len(names) != 3:
        with pytest.raises(AssertionError):
            compare_lists_of_string(li_1, li_2, li_3, names=names)
    else:
        chk = compare_lists_of_string(li_1, li_2, li_3, names=names)
        assert chk == exp


def test_flatten():
    """Test 'flatten' function."""
    inputs = (
        ([], []),
        ([0], [0]),
        ([1], [1]),
        ([{4}], [{4}]),
        ([[0], [1], [2]], [0, 1, 2]),
        ([[0], 1, [2]], [0, 1, 2]),
        ([["ab"], "cd", ["ef"]], ["ab", "cd", "ef"]),
    )
    for to_check, expected in inputs:
        chk = list(flatten(to_check))
        assert chk == expected


def test_ensure_list():
    """Test 'ensure_flat_list' function."""
    inputs = (
        (None, []),
        (False, [False]),
        ([], []),
        (0, [0]),
        (1, [1]),
        ([0], [0]),
        ([1], [1]),
        ("string", ["string"]),
        (["string"], ["string"]),
        ([["a", "b"]], [["a", "b"]]),
        ([(1,)], [(1,)]),
        ([{4}], [{4}]),
    )
    for to_check, expected in inputs:
        chk = list(ensure_list(to_check))
        assert chk == expected


def test_ensure_flat_list():
    """Test 'ensure_list' function."""
    inputs = (
        (None, []),
        ([], []),
        ("string", ["string"]),
        (["string"], ["string"]),
        ([0], [0]),
        ([1], [1]),
        ([{4}], [{4}]),
        ([[0], [1], [2]], [0, 1, 2]),
        ([[0], 1, [2]], [0, 1, 2]),
        ([["ab"], "cd", ["ef"]], ["ab", "cd", "ef"]),
    )
    for to_check, expected in inputs:
        chk = list(ensure_flat_list(to_check))
        assert chk == expected


@pytest.mark.parametrize(
    "o, n, exp",
    [
        ([], 2, True),
        ([(1, 2), (3, 4)], 2, True),
        ([(1, 2), (3, 4, 5)], 2, False),
    ],
)
def test_ensure_nested_length(o: list, n: int, exp: bool):
    """Test 'ensure_nested_length' function."""
    chk = ensure_nested_length(o, n)
    assert chk == exp


def test_get_class_name():
    """Test the 'get_class_name' function."""
    # noqa: D202
    class A: ...  # noqa

    class AA: ...  # noqa

    class AA1: ...  # noqa

    class Test: ...  # noqa

    class TestClass: ...  # noqa

    li_tests = [
        [A, ("A", "a")],
        [AA, ("AA", "a_a")],
        [AA1, ("AA1", "a_a1")],
        [Test, ("Test", "test")],
        [TestClass, ("TestClass", "test_class")],
    ]

    for obj, exp in li_tests:
        assert get_class_name(obj) == exp
        assert get_class_name(obj()) == exp


class TestSymmetricDifferences:
    """Test for 'get_symmetric_differences_in_sets' function."""

    def test_empty_sets(self):
        """Test no input."""
        assert get_symmetric_differences_in_sets() == set()

    def test_single_set(self):
        """Test a single set."""
        assert get_symmetric_differences_in_sets({1, 2}) == set()

    def test_wrong_inputs(self):
        """Test the wrong input type."""
        with pytest.raises(TypeError):
            get_symmetric_differences_in_sets([1, 2, 3])

    def test_multiple_sets(self):
        """Test 'get_symmetric_differences_in_sets' function with multiple sets."""
        s1 = {1, 2, 3}
        s2 = {2, 3, 4}
        s3 = {3, 4, 5}
        assert get_symmetric_differences_in_sets(s1, s2, s3) == {1, 2, 4, 5}


@pytest.mark.parametrize(
    "o, t, exp",
    [
        ([], int, True),
        ([], str, True),
        ([1, 1, 2], int, True),
        ([1, 1, 2], str, False),
        ([1, 1, "2"], int, False),
        ([1, 1, "2"], str, False),
        (["1", "2"], str, True),
        (["1", "2"], int, False),
    ],
)
def test_is_list_uniform(o: list, t: type, exp: bool):
    """Test 'is_list_uniform' function."""
    chk = is_list_uniform(o, t)
    assert chk == exp


class TestSelectColumns:
    """Unit-test for 'select_columns' function."""

    @pytest.mark.parametrize("allow", [True, False])
    def test_excess(self, allow):
        """Test the functionality of 'allow_excess_columns' in 'select_columns'."""
        input_cols = [f"{x}{y}" for x, y in itertools.product("cd", range(5))]

        if allow:
            # Excess exception disabled
            select_columns(
                input_cols,
                columns=["c1", "c2", "c3", "hello"],
                allow_excess_columns=True,
            )
        else:
            # Test exceptions
            with pytest.raises(AssertionError):
                select_columns(input_cols, columns=["c1", "c2", "c3", "c0", "c0"])

            with pytest.raises(AssertionError):
                select_columns(input_cols, columns=["c1", "c2", "c3", "hello"])

    def test_no_arguments(self):
        """Test 'select_columns' function without arguments."""
        chk = select_columns(["1", "a", "b2"])
        assert chk == []

    @pytest.mark.parametrize(
        "columns, regex, glob, startswith, endswith",
        [
            ("a", None, None, "x", None),
            (None, "a", None, "x", None),
            (None, None, "*", "x", None),
            ("a", None, None, None, "x"),
            (None, "a", None, None, "x"),
            (None, None, "*", None, "x"),
            (None, None, None, "x", "x"),
        ],
    )
    def test_too_many_parameters(self, columns, regex, glob, startswith, endswith):
        """Test 'select_columns' with 'startswith' / 'endswith' and another parameter'."""
        with pytest.raises(AssertionError):
            select_columns(
                ["X"],
                columns=columns,
                regex=regex,
                glob=glob,
                startswith=startswith,
                endswith=endswith,
            )

    @pytest.mark.parametrize(
        "startswith, exp",
        [
            ("z", []),
            ("a", ["a_1", "a_2"]),
            ("b", ["b_1", "b_2"]),
            (["b"], ["b_1", "b_2"]),
            (["a", "b"], ["a_1", "a_2", "b_1", "b_2"]),
        ],
    )
    def test_startswith(self, startswith, exp):
        """Test 'select_columns' with 'startswith' parameter'."""
        input_columns = ["a_1", "a_2", "b_1", "b_2", "c_1", "c_2"]
        chk = select_columns(
            input_columns,
            startswith=startswith,
        )
        assert chk == exp

    @pytest.mark.parametrize(
        "endswith, exp",
        [
            ("0", []),
            ("1", ["a_1", "b_1", "c_1"]),
            (["1"], ["a_1", "b_1", "c_1"]),
            ("2", ["a_2", "b_2", "c_2"]),
            (["1", "2"], ["a_1", "a_2", "b_1", "b_2", "c_1", "c_2"]),
        ],
    )
    def test_endswith(self, endswith, exp):
        """Test 'select_columns' with 'endswith' parameter'."""
        input_columns = ["a_1", "a_2", "b_1", "b_2", "c_1", "c_2", "c_3"]
        chk = select_columns(
            input_columns,
            endswith=endswith,
        )
        assert chk == exp

    @pytest.mark.parametrize("columns", (None, ["c1", "c2"]))
    @pytest.mark.parametrize("regex", (None, "c[034]"))
    @pytest.mark.parametrize("glob", (None, "*", "d*"))
    def test_columns_regex_glob(
        self,
        columns: Optional[List[str]],
        regex: Optional[str],
        glob: Optional[str],
    ):
        """Test 'select_columns' with the parameters 'columns', 'regex' and 'glob'."""
        input_cols = [f"{x}{y}" for x, y in itertools.product("cd", range(5))]

        cols = set(columns or [])
        regex_cols = {"c0", "c3", "c4"} if regex else set()

        set_d_start = {x for x in input_cols if x.startswith("d")}

        if glob is None:
            glob_cols = set()
        elif glob == "*":
            glob_cols = set(input_cols)
        else:
            glob_cols = set_d_start

        glob_cols -= regex_cols | cols

        expected = itertools.chain(
            columns or [],
            (i for i in input_cols if i in regex_cols),
            (i for i in input_cols if i in glob_cols),
        )

        actual = select_columns(
            input_cols,
            columns=columns,
            regex=regex,
            glob=glob,
        )
        assert list(expected) == actual


class TestSplitStringInChunks:
    """Unit-test for 'split_string_in_chunks' function."""

    def test_short_string(self):
        """Short string."""
        assert split_string_in_chunks("short", 10) == ["short"]

    def test_exact_length_string(self):
        """Check length of string."""
        chk = split_string_in_chunks("exact_length", 12)
        assert chk == ["exact_length"]

    def test_long_string(self):
        """Long string."""
        s = "This is a very long string that needs to be split."
        chk = split_string_in_chunks(s, 10)
        exp = ["This is a", "very long", "string", "that needs", "to be", "split."]
        assert chk == exp

    def test_long_string_without_blacks(self):
        """Long string without blank spaces."""
        s = "This is a very long string that needs to be split.".replace(" ", "")
        chk = split_string_in_chunks(s, 10)
        assert max(len(i) for i in chk) == 10

    def test_empty_string(self):
        """Empty string."""
        assert split_string_in_chunks("", 5) == []

    def test_max_length_zero(self):
        """Wrong length."""
        with pytest.raises(ValueError):
            split_string_in_chunks("test", 0)


def test_truncate_long_string():
    """Test 'truncate_long_string' function."""
    max_len = 20
    n_chk = 3
    for _ in range(100):
        n = random.randint(0, 100)
        s = "".join(random.choice(ascii_lowercase) for _ in range(n))
        chk: str = truncate_long_string(s, max_len)
        if n > max_len:
            assert "..." in chk
            assert len(chk) <= max_len
            assert s[:n_chk] == chk[:n_chk]
            assert s[-n_chk:] == chk[-n_chk:]
        else:
            assert chk == s


def test_validate_regex_pattern():
    """Test 'validate_regex_pattern'."""
    assert validate_regex_pattern(r"\d{3}-\d{2}-\d{4}")


def test_validate_regex_pattern_error():
    """Test 'validate_regex_pattern' exception."""
    with pytest.raises(ValueError):
        validate_regex_pattern(r"(unclosed bracket")
