"""Behavioral tests for competition micro ROC AUC and input validation."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import roc_auc_score

from sintef_project.kaggle_metric import ParticipantVisibleError, score


class MetricTests(unittest.TestCase):
    """Check ID alignment, pooled-label ranking, and malformed inputs."""

    def setUp(self) -> None:
        self.solution = pd.DataFrame({"Run No": [10, 20], "a": [0, 1], "b": [1, 0]})
        self.submission = pd.DataFrame(
            {"Run No": [10, 20], "a": [0.1, 0.4], "b": [0.8, 0.7]}
        )

    def test_expected_micro_auc_differs_from_macro(self) -> None:
        # Three of four positive-negative comparisons are correctly ordered.
        self.assertEqual(score(self.solution, self.submission), 0.75)
        self.assertEqual(
            roc_auc_score(
                self.solution[["a", "b"]], self.submission[["a", "b"]], average="macro"
            ),
            1.0,
        )

    def test_aligns_shuffled_rows_and_columns(self) -> None:
        shuffled = self.submission.iloc[::-1][["b", "Run No", "a"]]
        self.assertEqual(score(self.solution, shuffled), 0.75)

    def test_accepts_full_submission_for_solution_subset(self) -> None:
        full = pd.concat(
            [self.submission, pd.DataFrame({"Run No": [30], "a": [0.2], "b": [0.3]})],
            ignore_index=True,
        )
        full["unused"] = 0.0
        self.assertEqual(score(self.solution, full), 0.75)

    def test_index_ids_and_custom_id_column(self) -> None:
        self.assertEqual(
            score(self.solution.set_index("Run No"), self.submission.set_index("Run No")),
            0.75,
        )
        self.assertEqual(
            score(
                self.solution.rename(columns={"Run No": "id"}),
                self.submission.rename(columns={"Run No": "id"}),
                "id",
            ),
            0.75,
        )

    def test_single_row_and_single_label(self) -> None:
        self.assertEqual(score(self.solution.iloc[[0]], self.submission), 1.0)
        self.assertEqual(
            score(self.solution[["Run No", "a"]], self.submission[["Run No", "a"]]), 1.0
        )

    def test_ties_and_constant_label_columns(self) -> None:
        truth = pd.DataFrame({"a": [0, 0], "b": [1, 1]})
        tied = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]})
        self.assertEqual(score(truth, tied), 0.5)

    def test_does_not_mutate_inputs(self) -> None:
        truth, prediction = self.solution.copy(deep=True), self.submission.copy(deep=True)
        score(self.solution, self.submission)
        assert_frame_equal(self.solution, truth)
        assert_frame_equal(self.submission, prediction)

    def test_rejects_missing_ids_or_labels(self) -> None:
        for prediction in (self.submission.iloc[[0]], self.submission.drop(columns="b")):
            with self.subTest(prediction=prediction), self.assertRaises(ParticipantVisibleError):
                score(self.solution, prediction)

    def test_rejects_duplicate_ids_columns_and_missing_ids(self) -> None:
        malformed = [
            self.submission.assign(**{"Run No": [10, 10]}),
            self.submission.assign(**{"Run No": [10, np.nan]}),
            self.submission.rename(columns={"b": "a"}),
        ]
        for frame in malformed:
            with self.subTest(frame=frame):
                with self.assertRaises(ParticipantVisibleError):
                    score(self.solution, frame)
                with self.assertRaises(ParticipantVisibleError):
                    score(frame, self.submission)

    def test_rejects_invalid_probabilities(self) -> None:
        for value in (-0.1, 1.1, np.inf, -np.inf, np.nan, "0.5", 0.5 + 1j):
            with self.subTest(value=value), self.assertRaises(ParticipantVisibleError):
                score(self.solution, self.submission.assign(a=[value, value]))

    def test_rejects_invalid_labels(self) -> None:
        for value in (-1, 2, 0.5, np.nan, np.inf, "1", 1 + 1j):
            with self.subTest(value=value), self.assertRaises(ParticipantVisibleError):
                score(self.solution.assign(a=[value, value]), self.submission)

    def test_rejects_empty_inputs_and_undefined_auc(self) -> None:
        with self.assertRaises(ParticipantVisibleError):
            score(self.solution.iloc[:0], self.submission)
        with self.assertRaises(ParticipantVisibleError):
            score(self.solution, self.submission.iloc[:0])
        with self.assertRaises(ParticipantVisibleError):
            score(self.solution.assign(a=0, b=0), self.submission)
        with self.assertRaises(ParticipantVisibleError):
            score(self.solution, None)

    def test_nullable_numeric_missing_value_is_participant_error(self) -> None:
        prediction = self.submission.copy()
        prediction["a"] = pd.array([0.1, pd.NA], dtype="Float64")
        with self.assertRaises(ParticipantVisibleError):
            score(self.solution, prediction)


if __name__ == "__main__":
    unittest.main()
