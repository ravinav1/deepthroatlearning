"""Small real-CSV integration tests for forecast time and label alignment."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from sintef_project.features import HORIZON, build_features


class FeaturePipelineTests(unittest.TestCase):
    """Exercise the complete builder against explicit optimizer-input fixtures."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.kernel = self.root / "data/kernel"
        self.extended = self.root / "data/extended"
        self.kernel.mkdir(parents=True)
        self.extended.mkdir(parents=True)
        self.dates = pd.date_range("2015-01-01", periods=10 * 24, freq="h", tz="UTC")
        self.reservoirs = [f"R{i}" for i in range(17)]
        self.units = ["Plant_G1", "Plant_G2"]
        self.label_columns = [
            f"result_committed_{unit}_t{hour}"
            for unit in self.units for hour in range(HORIZON)
        ]
        # Deliberately retain nonchronological CSV order and arbitrary IDs.
        records = []
        for case, (run_id, start) in enumerate(((30, "2015-01-03"), (10, "2015-01-01"))):
            record = {"Run No": run_id, "starttime": start, "calculation_time": 1.0}
            record.update({
                column: int((case + j) % 3 == 0)
                for j, column in enumerate(self.label_columns)
            })
            records.append(record)
        pd.DataFrame(records).to_csv(self.kernel / "Unit_commitment_decisions.csv", index=False)
        pd.DataFrame({"Run No": [20], "starttime": ["2015-01-02"]}).to_csv(
            self.root / "prediction_mapping.csv", index=False
        )
        pd.DataFrame([{**{"Run No": 20}, **dict.fromkeys(self.label_columns, 0)}]).to_csv(
            self.root / "sample_submission.csv", index=False
        )
        topology = ["model:", "  reservoir:"]
        for reservoir in self.reservoirs:
            topology.extend([f"    {reservoir}:", "      max_vol: 1000"])
        topology.append("  plant:")
        (self.extended / "Tokke_Vinje_topology.yaml").write_text("\n".join(topology))
        self.write_series("Historical_day_ahead_price_2015_2025.csv", ["price"], 0)
        self.write_series("Historical_volume_2015_2024.csv", self.reservoirs, 100)
        self.write_series("Synthetic_water_value_2015_2024.csv", self.reservoirs, 500)
        self.write_series("Historical_inflow_1958_2025.csv", self.reservoirs, 1)
        self.write_series("Constraint_min_volume.csv", ["R0"], 0, extended=True)
        self.write_series("Constraint_min_flow.csv", ["R0"], 0, extended=True)

    def write_series(self, name: str, columns: list[str], offset: float, extended: bool = False) -> None:
        """Write hourly values whose expected lookup is easy to calculate."""
        values = {"date": self.dates}
        values.update({column: np.arange(len(self.dates)) + offset + i
                       for i, column in enumerate(columns)})
        pd.DataFrame(values).to_csv((self.extended if extended else self.kernel) / name, index=False)

    def test_case_hour_order_and_exact_forecast_inputs(self) -> None:
        x, labels, train, mapping, meta = build_features(self.root)
        self.assertEqual(len(x), 3 * HORIZON)
        self.assertEqual(train["Run No"].tolist(), [30, 10])
        self.assertEqual(mapping["Run No"].tolist(), [20])
        self.assertEqual(meta["units"], self.units)
        for case, start_hour in enumerate((48, 0, 24)):
            case_x = x.iloc[case * HORIZON:(case + 1) * HORIZON]
            np.testing.assert_array_equal(case_x["price"], start_hour + np.arange(HORIZON))
            np.testing.assert_array_equal(case_x["horizon_hour"], np.arange(HORIZON))
            np.testing.assert_array_equal(case_x["volume_R0"], np.full(HORIZON, start_hour + 100))
            np.testing.assert_array_equal(
                case_x["water_value_R0"], np.full(HORIZON, start_hour + HORIZON + 500)
            )
        source = pd.read_csv(self.kernel / "Unit_commitment_decisions.csv")
        for unit in self.units:
            columns = [f"result_committed_{unit}_t{hour}" for hour in range(HORIZON)]
            np.testing.assert_array_equal(labels[unit], source[columns].to_numpy().ravel())

    def test_future_volume_and_target_changes_do_not_change_features(self) -> None:
        expected, old_labels, *_ = build_features(self.root)
        path = self.kernel / "Historical_volume_2015_2024.csv"
        volumes = pd.read_csv(path)
        volumes.loc[~volumes.index.isin([0, 24, 48]), self.reservoirs] = -1000
        volumes.to_csv(path, index=False)
        path = self.kernel / "Unit_commitment_decisions.csv"
        training = pd.read_csv(path)
        training[self.label_columns] = 1 - training[self.label_columns]
        training.to_csv(path, index=False)
        actual, new_labels, *_ = build_features(self.root)
        assert_frame_equal(actual, expected)
        for unit in self.units:
            np.testing.assert_array_equal(new_labels[unit], 1 - old_labels[unit])

    def test_missing_forecast_timestamp_fails_instead_of_imputing(self) -> None:
        path = self.kernel / "Historical_day_ahead_price_2015_2025.csv"
        prices = pd.read_csv(path).drop(index=100)
        prices.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Missing price timestamps"):
            build_features(self.root)


if __name__ == "__main__":
    unittest.main()
