"""Shared constants."""

from __future__ import annotations

PACKAGE_NAME = "lora-header-inspector"
DEFAULT_INSPECTION_CSV = "lora_inspect_index.csv"
DEFAULT_MAX_HEADER_BYTES = 100 * 1024 * 1024  # official safetensors limit
MAX_HEADER_BYTES = DEFAULT_MAX_HEADER_BYTES

# Official safetensors dtype identifiers and their per-element byte sizes.
# Unknown dtypes produce a warning by default (a sloppy file can still be
# classified) and an error under strict layout checking (its nbytes cannot
# be verified).
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
ALLOWED_DTYPES = frozenset(DTYPE_SIZES)

# The safetensors schema bounds dimensions and offsets by a 48-bit size.
MAX_TENSOR_DIMENSION = 2**48
HASH_CHUNK_BYTES = 8 * 1024 * 1024

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
CONF_RANK = CONFIDENCE_RANK
CONFIDENCE_VALUES = tuple(CONFIDENCE_RANK)
STATUS_VALUES = ("ok", "ambiguous", "conflict", "error")

TRUTHY = frozenset({"true", "1", "yes", "y", "on"})
FALSY = frozenset({"false", "0", "no", "n", "off"})

WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
MODEL_LIKE_SUFFIXES = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
)
