# Edge cases for numpy string transfer to DAPHNE via FixedStr16 shared memory.

import numpy as np
import pandas as pd

from daphne.context.daphne_context import DaphneContext


LONG_STRING_ERROR = (
    "transferring a numpy array of strings longer than 15 bytes "
    "to DAPHNE via shared memory is not supported yet"
)
UNSUPPORTED_OBJECT_ERROR = (
    "object arrays transferred as FixedStr16 may only contain "
    "str, numpy.str_, bytes, numpy.bytes_, bytearray, memoryview, "
    "None, np.nan, pd.NA, or pd.NaT values"
)


def assert_raises(expected_exception, expected_message, func):
    try:
        func()
    except expected_exception as error:
        if str(error) != expected_message:
            raise AssertionError(
                f"expected error message {expected_message!r}, got {str(error)!r}"
            ) from error
        return
    raise AssertionError(f"expected {expected_exception.__name__}")


def print_matrix(matrix):
    matrix.print().compute()


dctx = DaphneContext()

# Exactly 15 bytes is the maximum payload for FixedStr16.
print_matrix(dctx.from_numpy(np.array([b"123456789012345"], dtype="S15"), shared_memory=True))
print_matrix(dctx.from_numpy(np.array([b"123456789012345"], dtype="S16"), shared_memory=True))

# Unicode input is accepted if its UTF-8 representation fits in 15 bytes.
print_matrix(dctx.from_numpy(np.array(["é" * 7], dtype=str), shared_memory=True))

# Object arrays cover the explicit Python-side conversion path.
object_values = np.array(
    [["plain", np.str_("npstr"), b"bytes"], [bytearray(b"ba"), memoryview(b"mv"), None]],
    dtype=object,
)
print_matrix(dctx.from_numpy(object_values, shared_memory=True))

# Null-like values use NumPy's direct fixed-width string conversion.
null_like_values = np.array([None, np.nan, pd.NA, pd.NaT], dtype=object)
print_matrix(dctx.from_numpy(null_like_values, shared_memory=True))

# The vectorized null conversion also applies inside otherwise mixed arrays.
mixed_null_like_values = np.array([["text", np.nan, pd.NA, pd.NaT]], dtype=object)
print_matrix(dctx.from_numpy(mixed_null_like_values, shared_memory=True))

# 16 bytes must be rejected because FixedStr16 needs one byte for null termination.
assert_raises(
    RuntimeError,
    LONG_STRING_ERROR,
    lambda: dctx.from_numpy(np.array([b"1234567890123456"], dtype="S16"), shared_memory=True),
)
assert_raises(
    RuntimeError,
    LONG_STRING_ERROR,
    lambda: dctx.from_numpy(np.array(["é" * 8], dtype=str), shared_memory=True),
)
assert_raises(
    RuntimeError,
    LONG_STRING_ERROR,
    lambda: dctx.from_numpy(np.array([b"1234567890123456"], dtype=object), shared_memory=True),
)
assert_raises(
    TypeError,
    UNSUPPORTED_OBJECT_ERROR,
    lambda: dctx.from_numpy(np.array([object()], dtype=object), shared_memory=True),
)

print("ok")
