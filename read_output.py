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

def get_signal_data(path):
    with open(path, "rb") as f:
        blob = f.read()

    header_end = blob.index(b"...\n") + 4
    header = yaml.safe_load(blob[:header_end])
    payload = blob[header_end:]

    # print("Header:", header)

    layout, record_size = infer_record_layout(header.get("data"))

    n_records = len(payload) // record_size
    payload = payload[: n_records * record_size]

    signal_meta = next((field for field in layout if field["name"] == "signal"), None)
    if signal_meta is None:
        raise ValueError("No 'signal' field found in header data description")

    signal_dtype = TYPE_NUMPY[signal_meta["dtype"]]
    signal_offset = signal_meta["offset"]
    signal_n_items = signal_meta["n_items"]

    if n_records == 0:
        return None

    signal_flat = np.ndarray(
        shape=(n_records, signal_n_items),
        dtype=signal_dtype,
        buffer=payload,
        offset=signal_offset,
        strides=(record_size, np.dtype(signal_dtype).itemsize),
    )

    samples = signal_flat.reshape((n_records, *signal_meta["dims"]))
    print("samples shape:", samples.shape, "dtype:", samples.dtype)
    return samples