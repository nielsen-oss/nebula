import narwhals as nw

__all__ = [
    "nw_count",
]

def nw_count(df: nw.DataFrame) -> int:
    df = df.collect() if isinstance(df, nw.LazyFrame) else df
    return df.shape[0]
