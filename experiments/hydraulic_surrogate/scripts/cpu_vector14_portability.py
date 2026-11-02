"""Byte-checked NPY bridge for the original numeric664 DataFrame and isolated XGB.

These helpers never construct competition features, read targets, or fit models.
The prospective fold runner must supply its independently verified row ordering.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np


def sha(path: Path) -> str:
    """Hash a complete immutable NPY file."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_dense_values(matrix, shape: tuple[int, int]) -> np.ndarray:
    """Read stored finite dense inputs without CSR.toarray losing signed-zero bits."""
    stored = matrix.get_data()
    rows, columns = shape
    if (stored.shape != shape or stored.nnz != rows * columns or
            not np.array_equal(stored.indptr, np.arange(rows + 1) * columns) or
            not np.array_equal(stored.indices, np.tile(np.arange(columns), rows))):
        raise ValueError('Finite dense native matrix structure changed')
    return stored.data.reshape(shape)


def verify_dmatrix_blocks(values: np.ndarray, names: list[str], xgb, block_rows: int = 2048) -> dict:
    """Require actual native DMatrix ingestion to preserve each float32 input byte."""
    if values.ndim != 2 or values.shape[1] != len(names) or values.dtype != np.dtype('<f4'):
        raise ValueError('Expected a two-dimensional little-endian float32 model input')
    if len(names) != len(set(names)) or block_rows < 1:
        raise ValueError('Invalid column names or block size')
    records = []
    for begin in range(0, len(values), block_rows):
        block = np.ascontiguousarray(values[begin:begin + block_rows])
        if not np.isfinite(block).all():
            raise ValueError('Original664 recipe requires finite features')
        native = xgb.DMatrix(block, feature_names=names, nthread=2)
        actual = native_dense_values(native, block.shape)
        if actual.dtype != block.dtype or actual.shape != block.shape or actual.tobytes() != block.tobytes():
            raise ValueError('Native DMatrix changed the actual float32 input bytes')
        records.append({'begin': begin, 'end': begin + len(block),
                        'float32_sha256': hashlib.sha256(block.tobytes()).hexdigest()})
    return {'xgboost': xgb.__version__, 'all_native_input_blocks_bitwise': True,
            'shape': list(values.shape), 'dtype': values.dtype.str, 'blocks': records}


def write_dataframe_rows(frame, rows: np.ndarray, path: Path, names: list[str], xgb,
                         block_rows: int = 2048) -> dict:
    """Export exact ordered rows and compare original DataFrame native ingestion.

    This is deliberately a low-level helper: callers own the frozen row IDs,
    samples, feature-source closure, target masks and execution authorization.
    """
    if path.exists() or frame.columns.tolist() != names or rows.ndim != 1:
        raise ValueError('Output exists or source feature/row schema differs')
    if rows.dtype.kind not in 'iu' or not len(rows) or np.any(rows < 0) or np.any(rows >= len(frame)):
        raise ValueError('Invalid row index vector')
    if len(np.unique(rows)) != len(rows) or block_rows < 1:
        raise ValueError('Duplicate rows or invalid block size')
    output = np.lib.format.open_memmap(path, mode='w+', dtype='<f4', shape=(len(rows), len(names)))
    block_records = []
    for begin in range(0, len(rows), block_rows):
        selected = frame.iloc[rows[begin:begin + block_rows]]
        block = np.ascontiguousarray(selected.to_numpy(dtype=np.float32))
        if not np.isfinite(block).all():
            raise ValueError('Original664 recipe requires finite features')
        original = native_dense_values(xgb.DMatrix(selected, feature_names=names, nthread=2), block.shape)
        if original.dtype != block.dtype or original.shape != block.shape or original.tobytes() != block.tobytes():
            raise ValueError('DataFrame native model input differs from portable float32 bytes')
        output[begin:begin + len(block)] = block
        block_records.append({'begin': begin, 'end': begin + len(block),
                              'float32_sha256': hashlib.sha256(block.tobytes()).hexdigest()})
    output.flush(); del output
    reopened = np.load(path, mmap_mode='r', allow_pickle=False)
    native = verify_dmatrix_blocks(reopened, names, xgb, block_rows)
    if native['blocks'] != block_records:
        raise ValueError('Saved NPY native inputs differ from the original DataFrame')
    return {'path': str(path), 'sha256': sha(path), 'shape': list(reopened.shape),
            'dtype': reopened.dtype.str, 'row_index_bytes_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
            'source_dtypes': [str(t) for t in frame.dtypes], 'feature_columns': names,
            'dataframe_and_npy_native_inputs_bitwise': True, 'native': native}
