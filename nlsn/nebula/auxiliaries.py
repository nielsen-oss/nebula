"""Auxiliaries functions that are not related to spark.

To avoid circular import, don't import anything from nebula into this file.
"""

import inspect
import re
from collections import Counter
from fnmatch import fnmatch
from itertools import chain
from typing import Callable, Generator, Iterable

__all__ = [
    "assert_allowed",
    "assert_at_least_one_non_null",
    "assert_at_most_one_args",
    "assert_only_one_non_none",
    "compare_lists_of_string",
    "ensure_flat_list",
    "ensure_list",
    "ensure_nested_length",
    "extract_kwarg_names",
    "flatten",
    "get_class_name",
    "get_symmetric_differences_in_sets",
    "is_list_uniform",
    "select_columns",
    "split_string_in_chunks",
    "truncate_long_string",
    "validate_keys",
    "validate_regex_pattern",
]


def assert_allowed(value, allowed: set, name: str) -> None:
    """Assert that the provided value is in the allowed set."""
    if value not in allowed:
        msg = f"'{name}' must be one of {allowed}, found: {value}"
        raise ValueError(msg)


def _assert_args(frame, args, func) -> tuple[int, str, str]:
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    full_names = string[string.find("(") + 1: -1].split(",")
    full_names = [i.strip() for i in full_names]

    s: int = 0
    names = []
    for el, name in zip(args, full_names):
        if func(el):
            continue
        s += 1
        names.append(name)

    names_joined = "', '".join(names)
    full_names_joined = "', '".join(full_names)
    full_names_joined = "{'" + full_names_joined + "'}"
    return s, names_joined, full_names_joined


def assert_at_least_one_non_null(*args):
    """Assert that at least one variable in '*args' is not None."""
    # Get the variable names in args
    frame = inspect.currentframe()
    s, _, full_names_joined = _assert_args(frame, args, lambda x: x is None)
    if s == 0:
        msg = f"At least one argument among {full_names_joined} must be non-null."
        raise AssertionError(msg)


def assert_at_most_one_args(*args):
    """Assert that at most, only one variable in '*args' is True."""
    # Get the variable names in args
    frame = inspect.currentframe()
    s, names_joined, full_names_joined = _assert_args(frame, args, lambda x: not x)

    if s > 1:
        msg = f"Only one among {full_names_joined} can be set. "
        msg += f"Found: '{names_joined}'."
        raise AssertionError(msg)


def assert_only_one_non_none(*args):
    """Assert that only one variable in '*args' is not None."""
    # Get the variable names in args
    frame = inspect.currentframe()
    s, names_joined, full_names_joined = _assert_args(frame, args, lambda x: x is None)
    if s == 0:
        msg = f"One and only one argument among {full_names_joined} must be non-null."
        raise AssertionError(msg)
    if s > 1:
        msg = f"Only one among {full_names_joined} can be non-null. Provided: '{names_joined}'."
        raise AssertionError(msg)


def compare_lists_of_string(
        *lists: list[str], names: list[str] | None = None
) -> list[str]:
    """Compare lists of strings and represent the differences in a tabular format.

     Args:
         *lists (list(str)):
             An indefinite number of lists of strings to compare.
         names (list(str) | None):
             Optional list representing the column names.
             If this list is provided it must have as many elements as
             the number of the provided 'lists'.

     Returns (list(str)):
         A list of strings representing the differences in a tabular format.

    Raises:
     AssertionError:
         If the number of 'names' provided is not equal to the number of *lists.


     Example:
         >>> li_1 = ['aa', 'cc', 'dd']
         >>> li_2 = ['aa']
         >>> li_3 = ['bb', 'cc']
         >>> compare_lists_of_string(li_1, li_2, li_3, names=['Li_1', 'Li_2', 'Li_3'])
         Output:
         ['Li_1 | Li_2 | Li_3',
          '-------------------',
          'aa   | aa   | ##',
          '##   | ##   | bb',
          'cc   | ##   | cc',
          'dd   | ##   | ##']
    """
    chained: list[str] = list(chain(*lists))

    max_len: int = max(len(i) for i in chained)
    n_cols: int = len(lists)
    if names:
        if len(names) != n_cols:
            msg = f'"lists" ({n_cols}) and "names" '
            msg += f"({len(names)}) must have the same length"
            raise AssertionError(msg)
        max_len_headers = max(len(i) for i in names)
        max_len = max(max_len_headers, max_len)

    sorted_set: list[str] = sorted(set(chained))
    n_rows: int = len(sorted_set)

    d_order: dict[str, int] = {v: i for i, v in enumerate(sorted_set)}
    # d_len = {v: len(v) for v in sorted_set}
    d_rows: dict[int, list[str]] = {i: [] for i in range(n_rows)}

    for li in lists:
        for el in sorted_set:
            i_rows: int = d_order[el]
            # len_ = d_len[el]
            if el in li:
                v = el
            else:
                v = "#" * max_len
            d_rows[i_rows].append(v)

    ret: list[str] = []

    if names:
        li_h_format = [i.ljust(max_len) for i in names]
        h_str = " | ".join(li_h_format)
        ret.append(h_str)
        ret.append("-" * ((n_cols * max_len) + (3 * (n_cols - 1))))

    for _, li_row in sorted(d_rows.items()):
        li_row_format = [i.ljust(max_len) for i in li_row]
        row_str = " | ".join(li_row_format)
        ret.append(row_str)

    return ret


def _is_list_or_tuple(o):
    """True if is <list> or <tuple>, False otherwise.

    There are many iterable types, <str>, <bytes>, <array>, <dict>, <set>, but
    here only lists and tuples are taken into account.
    """
    return isinstance(o, (list, tuple))


def ensure_list(o) -> list:
    """Ensure the output data type is a list.

    Input cases:
    - list: the output is the input untouched
    - None: the output is an empty list []
    - anything else: the input is enclosed in a list [x]
    """
    if o is None:
        return []
    if isinstance(o, list):
        return o
    return [o]


def ensure_flat_list(o) -> list:
    """Ensure the output data type is a flat list.

    Input cases:
    - list | tuple: the output will be flattened if it is a nested iterable
    - None: the output is an empty list []
    - anything else: the input is enclosed in a list [x]

    Returns:
        object:
    """
    if o is None:
        return []
    if _is_list_or_tuple(o):
        return list(flatten(o))
    return [o]


def ensure_nested_length(o: Iterable, n: int) -> bool:
    """Ensure that each element in the iterable 'o' has a length equal to 'n'.

    >>>ensure_nested_length([(1, 2), (3, 4)])
    True

    >>>ensure_nested_length([(1, 2), (3, 4, 5)])
    False

    Args:
        o (Iterable): The iterable to check.
        n (int): The desired length for each element in 'o'.

    Returns:
        bool: True if all elements in 'o' have length 'n', False otherwise.
    """
    return all(len(i) == n for i in o)


def extract_kwarg_names(o: Callable | type) -> list[str]:
    """Extracts the names of keyword-only arguments from the given function or class.

    Args:
        o (callable | type):
            The function or class to extract keyword-only argument names from.

    Returns (list(str)):
        Keyword-only arguments.
    """
    params = inspect.signature(o).parameters
    return [k for k, v in params.items() if v.kind.name == "KEYWORD_ONLY"]


def flatten(lst: Iterable) -> Generator:
    """Flatten a list.

    Args:
        lst: (list)
            List to flatten.
    Returns: (list)
        Flat list.
    """
    for el in lst:
        if _is_list_or_tuple(el):
            yield from flatten(el)
        else:
            yield el


def get_class_name(o) -> tuple[str, str]:
    """Get the name of the class or of the instantiated object.

    Convert it to snake-case and return both.

    Args:
        o: (class | object)

    Returns: (str, str)
        e.g.: (ThisClass, this_class)
    """
    camel_name = getattr(o, "__name__", o.__class__.__name__)
    snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_name).lower()
    return camel_name, snake_name


def get_symmetric_differences_in_sets(*list_sets: set) -> set:
    """Computes the symmetric difference of multiple sets.

    Args:
        *list_sets: Variable number of sets.

    Returns:
        set: Symmetric difference of the input sets.
    """
    if not list_sets:
        return set()

    if not all(isinstance(i, set) for i in list_sets):
        raise TypeError("The input must be a <list> or a <tuple> of <sets>.")

    if len(list_sets) < 2:
        return set()

    n = len(list_sets)

    count = Counter()
    for s in list_sets:
        count.update(s)

    return {k for k, v in count.items() if v != n}


def is_list_uniform(o: Iterable, t: type) -> bool:
    """Check if all elements in the iterable `o` are of the same type `t`.

    If an empty list is provided, the output will always be True.

    Args:
        o (Iterable): Iterable object to check.
        t (type): Type to check for uniformity, like str | int.

    Returns (bool):
        True if the list is empty or all elements are of type `t`,
        False otherwise.
    """
    return all(isinstance(i, t) for i in o)


def select_columns(
        input_columns: list[str] | None,
        *,
        columns: list[str] | None = None,
        regex: str | None = None,
        glob: str | None = None,
        startswith: str | Iterable[str] | None = None,
        endswith: str | Iterable[str] | None = None,
        allow_excess_columns: bool = False,
) -> list[str]:
    """Select a subset of columns given certain conditions.

    Given a data frame, a list of column names (optional),
    and a regex pattern (optional), select a subset of column
    names from the data frame, preserving their original order of
    the columns within that data frame.

    Args:
        input_columns (Iterable(str)):
            List of columns from which it will select a subset (usually df.columns).
        columns (Iterable(str) | None):
            A list of columns to select. Defaults to None.
        regex (str | None):
            A regular expression to select columns with. Defaults to None.
        glob (str | None):
            A glob (bash-like pattern) expression to select columns with.
            Defaults to None.
        startswith (str | Iterable(str) | None):
            Select all the columns whose names start with the provided
            string(s). Defaults to None.
        endswith (str | Iterable(str) | None):
            Select all the columns whose names end with the provided
            string(s). Defaults to None.
        allow_excess_columns (bool):
            Whether to allow columns that are not contained in the
            dataframe (raises AssertionError by default).
            Default to False.

    Returns:
        list(str): A subset of columns with the following structure:
           - All columns that are in `columns`, with the order of `columns`
           - All columns that match `regex`, preserving the original order
           - All columns that match `glob`, preserving the original order
    Raises:
        AssertionError: If `columns` has repeating elements
        AssertionError: If `columns` have elements not present in
            `input_columns` and `allow_excess_columns` is False
    """
    column_list = ensure_list(columns)

    # Check for repeating columns
    repeating_columns = {c for c, f in Counter(column_list).items() if f > 1}
    if repeating_columns:
        # There can be no repeating columns
        msg_rep = "The column list must not contain repeating strings, "
        msg_rep += f"but columns {repeating_columns} appear more than once"
        raise AssertionError(msg_rep)

    # Check for excess columns
    if not allow_excess_columns:
        not_found_columns = set(column_list) - set(input_columns)
        if not_found_columns:
            raise AssertionError(f"Columns not found in input: {not_found_columns}")

    # Handle startswith and endswith
    if startswith or endswith:
        assert_only_one_non_none(columns, regex, glob, startswith, endswith)
        if startswith:
            start: tuple[str] = tuple(ensure_list(startswith))
            return [i for i in input_columns if i.startswith(start)]
        else:
            end: tuple[str] = tuple(ensure_list(endswith))
            return [i for i in input_columns if i.endswith(end)]

    # Handle regex and glob
    if not (regex or glob):
        return column_list

    pattern = re.compile(regex or "a^")
    glob_expr = glob or ""

    columns_seen: set[str] = set()

    def _regex_matches(_c):
        return bool(pattern.search(_c)) and _c not in columns_seen

    def _glob_matches(_c):
        return fnmatch(_c, glob_expr) and _c not in columns_seen

    ret: list[str] = column_list[:]
    columns_seen.update(column_list)

    # ---- Appending regex
    re_match = list(filter(_regex_matches, input_columns))
    ret.extend(re_match)
    columns_seen.update(re_match)

    # ---- Appending glob
    ret.extend(filter(_glob_matches, input_columns))

    return ret


def split_string_in_chunks(long_string: str, limit: int = 30) -> list[str]:
    """Split a long string in chunks."""
    if limit <= 0:
        raise ValueError("'limit' must be greater than or equal to 0.")
    splits = long_string.split()
    short_splits = []
    for split in splits:
        if len(split) <= limit:
            short_splits.append(split)
        else:
            substrings = [split[i: i + limit] for i in range(0, len(split), limit)]
            short_splits.extend(substrings)

    tot = 0
    ret = []
    substring = []
    for split in short_splits:
        len_ = len(split)
        if (tot + len_) < limit:
            substring.append(split)
            tot += len_
        else:
            if len(substring) == 1:
                ret.append(" " + substring[0])
            else:
                ret.append(" ".join(substring))
            tot = len_
            substring = [split]

    if len(substring) == 1:
        ret.append(" " + substring[0])
    else:
        ret.append(" ".join(substring))

    return [i.strip() for i in ret if i.split()]


def truncate_long_string(s: str, w: int) -> str:
    """Truncates a long string to fit within a specified width.

    Args:
        s (str): The input string to be truncated.
        w (int): The maximum width for the truncated string.

    Returns (str):
        The truncated string. If the original string is shorter than
        the specified width, it is returned unchanged.

    Example:
    >>> truncate_long_string("This is a long string", 10)
    'Th ... ng'
    """
    half_w = w // 2 - 3

    if len(s) > w:
        return s[:half_w] + " ... " + s[-half_w:]
    else:
        return s


def validate_keys(
        name: str,
        data: dict | set,
        *,
        mandatory: set[str] | None = None,
        optional: set[str] | None = None,
) -> None:
    """Validate that a dictionary has required keys and no unexpected ones.

    Args:
        name: Name of the parameter (for error messages)
        data: Dictionary or set of keys to validate
        mandatory: Required keys that must be present
        optional: Optional keys that are allowed (defaults to empty set)

    Raises:
        KeyError: If required keys are missing or unknown keys are present

    Example:
        >>> validate_keys(
        ...     "config",
        ...     {"alias": "x", "value": 1},
        ...     mandatory={"alias"},
        ...     optional={"value", "cast"}
        ... )
    """
    mandatory = mandatory or set()
    optional = optional or set()
    all_keys = mandatory | optional
    keys = set(data)

    if not keys.issubset(all_keys):
        excess = keys - all_keys
        raise KeyError(
            f"'{name}' keys must be a subset of {all_keys}. "
            f"Unknown key(s): {excess}"
        )

    if not mandatory.issubset(keys):
        missing = mandatory - keys
        raise KeyError(
            f"'{name}' must contain the key(s): {mandatory}. "
            f"Missing: {missing}"
        )


def validate_regex_pattern(pattern: str) -> bool:
    """Validate a regex pattern.

    Args:
        pattern: (str)
            Regex pattern to validate.

    Returns: (bool)
        True if 'pattern' is valid, False otherwise.
    """
    try:
        re.fullmatch(pattern, "")
        return True
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {str(e)}") from e
