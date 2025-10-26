"""Pipeline exceptions module."""

from nlsn.nebula.backend_util import HAS_SPARK

if HAS_SPARK:
    from py4j.protocol import Py4JJavaError
    from pyspark.sql.utils import CapturedException

__all__ = [
    "PipelineError",
    "raise_pipeline_error",
]


class PipelineError(Exception):
    pass


def raise_pipeline_error(e: Exception, msg: str) -> None:
    """Handle pipeline errors.

    Manageable exceptions:
    - Py4JJavaError (low-level spark error)
    - CapturedException (low-level spark error, like: AnalysisException, ...)
    - All the native Python exceptions

    Args:
        e (Exception): Exception raised.
        msg (str): Custom message to add.

    Raises:
        Original exception adding the custom message
    """
    # if len(e.args) >= 1:
    #     e.args = (e.args[0] + "\n" + msg,) + e.args[1:]
    # else:
    #     e.args = ("\n" + msg,)

    if HAS_SPARK and isinstance(e, Py4JJavaError):
        e.errmsg = msg + "\n" + e.errmsg
        raise e

    if HAS_SPARK and isinstance(e, CapturedException):
        e.desc = msg + "\n" + e.desc
        raise e

    raise type(e)(f"{e}\n{msg}")  # .with_traceback(sys.exc_info()[2])
