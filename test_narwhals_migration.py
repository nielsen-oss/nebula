#!/usr/bin/env python3
"""Test script to verify Narwhals migration works."""

import pandas as pd
import polars as pl
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("test").getOrCreate()
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

from nlsn.nebula.shared_transformers.columns import DropColumns, RenameColumns, SelectColumns

# Test data
data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}

def test_drop_columns():
    print("Testing DropColumns...")
    
    # Pandas
    df_pd = pd.DataFrame(data)
    transformer = DropColumns(columns=["b"])
    result_pd = transformer.transform(df_pd)
    print(f"Pandas result columns: {list(result_pd.columns)}")
    
    # Polars
    df_pl = pl.DataFrame(data)
    result_pl = transformer.transform(df_pl)
    print(f"Polars result columns: {result_pl.columns}")
    
    if SPARK_AVAILABLE:
        # Spark
        df_spark = spark.createDataFrame(pd.DataFrame(data))
        result_spark = transformer.transform(df_spark)
        print(f"Spark result columns: {result_spark.columns}")

def test_rename_columns():
    print("\nTesting RenameColumns...")
    
    # Pandas
    df_pd = pd.DataFrame(data)
    transformer = RenameColumns(mapping={"a": "x", "b": "y"})
    result_pd = transformer.transform(df_pd)
    print(f"Pandas result columns: {list(result_pd.columns)}")
    
    # Polars
    df_pl = pl.DataFrame(data)
    result_pl = transformer.transform(df_pl)
    print(f"Polars result columns: {result_pl.columns}")

def test_select_columns():
    print("\nTesting SelectColumns...")
    
    # Pandas
    df_pd = pd.DataFrame(data)
    transformer = SelectColumns(columns=["a", "c"])
    result_pd = transformer.transform(df_pd)
    print(f"Pandas result columns: {list(result_pd.columns)}")
    
    # Polars
    df_pl = pl.DataFrame(data)
    result_pl = transformer.transform(df_pl)
    print(f"Polars result columns: {result_pl.columns}")

if __name__ == "__main__":
    test_drop_columns()
    test_rename_columns()
    test_select_columns()
    print("\nAll tests completed!")