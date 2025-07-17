"""Example of custom transformers."""

from nlsn.nebula.storage import nebula_storage as ns


class Custom_3:
    @staticmethod
    def transform(df):
        """Public static method transform."""
        return df


class SetToNebulaStorage:
    @staticmethod
    def transform(df):
        """Set a value to nebula storage."""
        ns.set("this_key", 10)
        return df


class ReadFromNebulaStorage:
    @staticmethod
    def transform(df):
        """Retrieve a value from nebula storage."""
        value = ns.get("this_key")
        print(f"read: {value}")
        return df
