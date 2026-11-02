"""Uniformly average aligned round5 prediction files, without fitting weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    assert len(args.inputs) >= 2 and len(set(args.inputs)) == len(args.inputs)
    assert not Path(args.output).exists() and not Path(args.report).exists()
    frames = [pd.read_csv(path, float_precision="round_trip") for path in args.inputs]
    first = frames[0]
    assert first.columns[0] == "Run No" and first.shape[1] == 2353 and first["Run No"].is_unique
    arrays = []
    for frame in frames:
        assert frame.columns.equals(first.columns) and frame["Run No"].equals(first["Run No"])
        values = frame.iloc[:, 1:].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        arrays.append(values)
    result = first.copy()
    result.iloc[:, 1:] = np.mean(arrays, axis=0)
    result.to_csv(args.output, index=False, mode="x")
    report = {"method": "Uniform probability mean; no weight fitting or seed selection.",
              "sources": [{"path": path, "sha256": digest(path), "weight": 1 / len(frames)} for path in args.inputs],
              "output": args.output, "output_sha256": digest(args.output),
              "cases": len(first), "outputs_per_case": 2352}
    with open(args.report, "x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Averaged {len(frames)} files into {args.output}")


if __name__ == "__main__":
    main()
