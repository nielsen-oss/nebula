"""Unit-test for TimeOfDay."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import TimeOfDay


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("col_1", StringType(), True),
        StructField("exp_hour", IntegerType(), True),
        StructField("exp_minute", IntegerType(), True),
        StructField("exp_second", IntegerType(), True),
        StructField("exp_hour_start", IntegerType(), True),
        StructField("exp_minute_start", IntegerType(), True),
        StructField("exp_second_start", IntegerType(), True),
    ]
    data = [
        ("2023-09-19 08:45:30", 8, 45, 30, 8, 525, 31530),
        ("2023-09-19 15:30:15", 15, 30, 15, 15, 930, 55815),
        (None, None, None, None, None, None, None),
    ]

    ret = (
        spark.createDataFrame(data, schema=StructType(fields))
        .withColumn("col_2", F.to_timestamp("col_1", format="yyyy-MM-dd HH:mm:ss"))
        .persist()
    )
    return ret


@pytest.mark.parametrize("from_start", [True, False])
@pytest.mark.parametrize("output_col", ["col_1", "col_2", "exp"])
@pytest.mark.parametrize("input_col", ["col_1", "col_2"])
@pytest.mark.parametrize("unit", ["hour", "minute", "second"])
def test_time_of_day(df_input, unit, input_col, output_col, from_start):
    """Unit-test TimeOfDay."""
    t = TimeOfDay(
        input_col=input_col, output_col=output_col, unit=unit, from_start=from_start
    )
    df_chk = t.transform(df_input)

    col_exp: str = f"exp_{unit}"
    col_exp += "_start" if from_start else ""

    cond = F.col(output_col) != F.col(col_exp)
    cond |= F.col(output_col).isNull() & F.col(col_exp).isNotNull()
    cond |= F.col(output_col).isNotNull() & F.col(col_exp).isNull()
    n_diff: int = df_chk.filter(cond).count()
    assert n_diff == 0
