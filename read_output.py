import os
import struct
import re

import numpy as np
import yaml


TYPE_FORMAT = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
}

TYPE_NUMPY = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
}

def parse_data_entry(entry):
    # Expected format: "name dtype (d0,d1,...)" or "name dtype (d0)"
    match = re.match(r"^(.+?)\s+(\w+)\s*\(([^)]*)\)\s*$", entry)
    if not match:
        raise ValueError(f"Unsupported data description entry: {entry}")

    name = match.group(1).strip()
    dtype = match.group(2).strip().lower()
    dims = [int(part.strip()) for part in match.group(3).split(",") if part.strip()]
    if not dims:
        raise ValueError(f"Missing dimensions in entry: {entry}")

    return name, dtype, dims

def item_count(dims):
    count = 1
    for d in dims:
        count *= d
    return count

def infer_record_layout(header_data):
    if not isinstance(header_data, list) or not header_data:
        raise ValueError("Header does not contain a valid 'data' description list")

    layout = []
    record_size = 0

    for entry in header_data:
        name, dtype, dims = parse_data_entry(entry)
        fmt = TYPE_FORMAT.get(dtype)
        if fmt is None:
            raise ValueError(f"Unsupported dtype in header entry: {dtype}")

        n_items = item_count(dims)
        byte_size = struct.calcsize("<" + fmt) * n_items
        layout.append(
            {
                "name": name,
                "dtype": dtype,
                "dims": dims,
                "n_items": n_items,
                "byte_size": byte_size,
                "offset": record_size,
            }
        )
        record_size += byte_size

    return layout, record_size


def _find_header_end(blob: bytes) -> int:
    # Binary output files start with a YAML header terminated by a document end marker.
    # Support both LF and CRLF line endings.
    for marker in (b"...\n", b"...\r\n"):
        idx = blob.find(marker)
        if idx != -1:
            return idx + len(marker)
    raise ValueError("Could not locate YAML header terminator ('...') in file")


def _extract_field(payload: bytes, layout: list, record_size: int, n_records: int, name: str):
    meta = next((field for field in layout if field["name"] == name), None)
    if meta is None:
        return None

    dtype = TYPE_NUMPY[meta["dtype"]]
    offset = meta["offset"]
    n_items = meta["n_items"]

    flat = np.ndarray(
        shape=(n_records, n_items),
        dtype=dtype,
        buffer=payload,
        offset=offset,
        strides=(record_size, np.dtype(dtype).itemsize),
    )

    # Reshape back to declared dimensions
    dims = meta["dims"]
    if dims == [1]:
        return flat[:, 0]
    return flat.reshape((n_records, *dims))

def get_signal_data(path, channel=0, timestamps=False):
    with open(path, "rb") as f:
        blob = f.read()

    header_end = _find_header_end(blob)
    header = yaml.safe_load(blob[:header_end])
    payload = blob[header_end:]

    # print("Header:", header)

    layout, record_size = infer_record_layout(header.get("data"))

    n_records = len(payload) // record_size

    if n_records == 0:
        return None, None
    
    payload = payload[: n_records * record_size]

    signal_meta = next((field for field in layout if field["name"] == "signal" or field["name"] == "scalar_data"), None)
    if signal_meta is None:
        raise ValueError("No 'signal' or 'scalar_data' field found in header data description")

    signal_dtype = TYPE_NUMPY[signal_meta["dtype"]]
    signal_offset = signal_meta["offset"]
    signal_n_items = signal_meta["n_items"]
    signal_flat = np.ndarray(
        shape=(n_records, signal_n_items),
        dtype=signal_dtype,
        buffer=payload,
        offset=signal_offset,
        strides=(record_size, np.dtype(signal_dtype).itemsize),
    )

    samples = signal_flat.reshape((n_records, *signal_meta["dims"]))
    if len(signal_meta["dims"]) == 2:
        samples = samples[:, 0, :]
    print(os.path.basename(path), "samples shape:", samples.shape, "dtype:", samples.dtype)

    if timestamps:
        source_ts = _extract_field(payload, layout, record_size, n_records, "source_ts")
        # hardware_ts = _extract_field(payload, layout, record_size, n_records, "hardware_ts")
        # Keep the return value extensible: callers can pull what they need.
        # ts = {
        #     "source_ts": source_ts,
        #     "hardware_ts": hardware_ts,
        # }
        return samples[:, channel], source_ts
    else:
        return samples[:, channel], None