"""Unit-test for ConcatColumns."""

from typing import Dict, List

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import ConcatColumns


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("col1", StringType(), True),
        StructField("col2", StringType(), True),
        StructField("col3", StringType(), True),
    ]

    # fmt: off
    data = [
        (1, "Alice", "Bob", "Charlie"),
        (2, "David", "Eva", "Frank"),
        (3, None, "Hank", None),
        (4, None, None, None),
    ]
    # fmt: on
    return spark.createDataFrame(data, schema=StructType(fields)).cache()


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("null_if_any_null", [True, False])
def test_concat_columns_base(df_input, drop: bool, null_if_any_null: bool):
    """Test ConcatColumns with basic configuration."""
    cols_to_concat = ["col1", "col2", "col3"]
    new_col_name = "new_col"
    separator = "-"

    t = ConcatColumns(
        cols_to_concat=cols_to_concat,
        new_col_name=new_col_name,
        separator=separator,
        null_if_any_null=null_if_any_null,
        drop_initial_cols=drop,
    )
    df_chk = t.transform(df_input)

    cols_chk = df_chk.columns
    cols_exp = df_input.columns[:] + [new_col_name]
    if drop:
        cols_exp = [i for i in cols_exp if i not in cols_to_concat]

    assert cols_chk == cols_exp

    # Use the 'idx' cols as the key to create a dictionary check vs expected
    # list of list -> [idx, list(str)]
    li_input = df_input.rdd.map(lambda x: [x[0], list(x[1:])]).collect()
    # li_input -> [
    #   [1, ['Alice', 'Bob', 'Charlie']],
    #   [2, ['David', 'Eva', 'Frank']],
    #   [3, [None, 'Hank', None]],
    #   [4, [None, None, None]]
    # ]

    li_chk = df_chk.select("idx", new_col_name).rdd.map(lambda x: x[:]).collect()
    # li_chk = [
    #   (1, 'Alice-Bob-Charlie'),
    #   (2, 'David-Eva-Frank'),
    #   (3, None),
    #   (4, None)
    # ]
    res_chk = dict(li_chk)

    li_exp = []

    if null_if_any_null:
        # If there is a None in the input rows the result must be null.
        for idx, li in li_input:
            if any(i is None for i in li):
                joined = None
            else:
                joined = separator.join(li)
            li_exp.append((idx, joined))

    else:
        # None are not taken into account.
        for idx, li in li_input:
            li_join = [i for i in li if i is not None]
            if li_join:
                joined = separator.join(li_join)
            else:
                # If a row contains None only, the result must be an empty string "".
                joined = ""
            li_exp.append((idx, joined))

    res_exp = dict(li_exp)

    assert res_exp == res_chk


def test_concat_columns_concat_strategy_error():
    """Test ConcatColumns with wrong input parameters."""
    concat_strategy = {
        "new_column_name": "new_col",
        "separator": "_",
        "strategy": [{"column": "col1"}, {"constant": "WE"}, {"constant": "WD"}],
    }
    with pytest.raises(AssertionError):
        ConcatColumns(
            concat_strategy=[concat_strategy],
            new_col_name="new_col",
        )


def test_concat_columns_strategy_error():
    """Test ConcatColumns with wrong concat strategies."""
    # fmt: off
    list_concat_strategy = [
        (TypeError, "invalid_strategy"),  # Wrong concat strategy
        (KeyError, [{
            'new_column_name': 'new_col',
            'separator': '_',
            'invalid_key': 'value'  # Wrong key
        }]),
        (ValueError, [{
            'new_column_name': 123,  # Should be a string
            'separator': '_',
            'strategy': [{'column': 'hh_id'}, {'constant': 'WE'}]
        }]),
        (ValueError, [{
            'new_column_name': "new_col",
            'separator': 123,  # Should be a string
            'strategy': [{'column': 'hh_id'}, {'constant': 'WE'}]
        }]),
        (ValueError, [{
            'new_column_name': 'new_col',
            'separator': '_',
            'strategy': [{'column': 'hh_id', 'constant': 'WE'}]  # Should have only one key
        }]),
        (KeyError, [{
            'new_column_name': 'new_col',
            'separator': '_',
            'strategy': [{'invalid_key': 'value'}]  # Wrong key
        }])
    ]
    # fmt: on

    for error_type, concat_strategy in list_concat_strategy:
        with pytest.raises(error_type):
            ConcatColumns(concat_strategy=concat_strategy)


def test_concat_columns_strategy(df_input):
    """Test ConcatColumns with concat strategies."""
    # fmt: off
    concat_strategy = [
        {
            "new_column_name": "new_col",
            "separator": "_",
            "strategy": [{"column": "col1"}, {"constant": "WE"}, {"constant": "WD"}],
        },
        {
            "new_column_name": "new_col_2",
            "separator": "$",
            "strategy": [
                {"column": "col2"},
                {"constant": "m5"},
                {"column": "col3"},
            ],
        },
    ]
    # fmt: on

    t = ConcatColumns(concat_strategy=concat_strategy)
    df_chk = t.transform(df_input)

    li_input = df_input.rdd.map(lambda x: x.asDict()).collect()
    # li_input -> [
    #     {'idx': 1, 'col1': 'Alice', 'col2': 'Bob', 'col3': 'Charlie'},
    #     {'idx': 2, 'col1': 'David', 'col2': 'Eva', 'col3': 'Frank'},
    #     {'idx': 3, 'col1': None, 'col2': 'Hank', 'col3': None},
    #     {'idx': 4, 'col1': None, 'col2': None, 'col3': None}
    # ]

    nd: dict
    for nd in concat_strategy:
        li_exp = []
        new_col_name = nd["new_column_name"]
        separator = nd["separator"]
        li_chk = df_chk.select("idx", new_col_name).rdd.map(lambda x: x[:]).collect()
        res_chk = dict(li_chk)

        strategy: List[Dict[str, str]] = nd["strategy"]

        row_input: dict
        for row_input in li_input:
            li_s = []
            joined = ""
            idx: int = row_input["idx"]
            for d in strategy:
                key, value = list(d.items())[0]
                if key == "column":
                    s = row_input[value]
                    if s:
                        li_s.append(s)
                elif key == "constant":
                    li_s.append(value)
                else:
                    raise ValueError("Wrong unit-test")

                joined = separator.join(li_s)
            li_exp.append((idx, joined))

        res_exp = dict(li_exp)
        assert res_exp == res_chk
