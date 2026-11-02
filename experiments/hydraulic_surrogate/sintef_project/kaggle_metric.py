"""
Custom multilabel ROC AUC metric for a binary multilabel classification problem.

This metric expects two DataFrames:
- submission: contains predicted probabilities for each label
- solution: contains binary ground truth labels for each label

The metric calculates micro-average ROC AUC across all labels.

Required columns:
- submission: same label columns as solution, containing probabilities for class 1
- solution: same label columns as submission, containing binary ground truth labels
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    """Raise this for participant-facing errors."""


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "Run No") -> float:
    """
    Compute micro-average ROC AUC for multilabel binary classification.

    Every solution row and label must be present in submission. Extra submission
    rows and columns are ignored, allowing a solution subset to be scored against
    a full submission. Rows and labels are aligned by ID and name, respectively.

    Args:
        solution (pd.DataFrame): Binary ground truth labels.
        submission (pd.DataFrame): Predicted probabilities for label 1.
        row_id_column_name: ID column, or the DataFrame index when absent.

    Returns:
        float: micro-average ROC AUC
    """

    for name, frame in (("Solution", solution), ("Submission", submission)):
        if not isinstance(frame, pd.DataFrame):
            raise ParticipantVisibleError(f"{name} must be a pandas DataFrame.")
        if frame.columns.has_duplicates:
            raise ParticipantVisibleError(f"{name} contains duplicate column names.")

    if row_id_column_name in solution.columns:
        solution = solution.set_index(row_id_column_name)
    if row_id_column_name in submission.columns:
        submission = submission.set_index(row_id_column_name)

    if solution.empty or submission.empty:
        raise ParticipantVisibleError("Solution and submission must not be empty.")

    for name, frame in (("Solution", solution), ("Submission", submission)):
        if isinstance(frame.index, pd.MultiIndex):
            raise ParticipantVisibleError(f"{name} must have a single row ID index.")
        if frame.index.has_duplicates:
            raise ParticipantVisibleError(f"{name} contains duplicate row IDs.")
        if frame.index.isna().any():
            raise ParticipantVisibleError(f"{name} contains missing row IDs.")

    missing_ids = solution.index.difference(submission.index)
    if not missing_ids.empty:
        raise ParticipantVisibleError(
            f"Submission is missing {len(missing_ids)} required ID(s), for example: {missing_ids[:5].tolist()}"
        )

    missing_labels = solution.columns.difference(submission.columns)
    if not missing_labels.empty:
        raise ParticipantVisibleError(
            f"Submission is missing {len(missing_labels)} required label column(s), "
            f"for example: {missing_labels[:5].tolist()}"
        )
    submission = submission.loc[solution.index, solution.columns]

    if not all(
        pd.api.types.is_numeric_dtype(solution[col])
        and not pd.api.types.is_complex_dtype(solution[col])
        for col in solution.columns
    ):
        raise ParticipantVisibleError("Solution columns must be numeric binary labels.")
    if not all(
        pd.api.types.is_numeric_dtype(submission[col])
        and not pd.api.types.is_complex_dtype(submission[col])
        for col in submission.columns
    ):
        raise ParticipantVisibleError("Submission columns must be numeric probabilities.")

    labels = solution.to_numpy(dtype=float, na_value=np.nan)
    probabilities = submission.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ParticipantVisibleError("Solution labels must be finite binary values (0 or 1).")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ParticipantVisibleError("Submission probabilities must be finite and in [0, 1].")
    if np.unique(labels).size != 2:
        raise ParticipantVisibleError("ROC AUC requires both positive and negative solution labels.")

    try:
        # Flattening also handles one-row or one-column multilabel subsets without
        # scikit-learn interpreting them as a multiclass classification problem.
        result = roc_auc_score(labels.ravel(), probabilities.ravel())
    except ValueError as exc:
        raise ParticipantVisibleError(f"Error during ROC AUC calculation: {exc}") from exc

    if not np.isfinite(result):
        raise ParticipantVisibleError("Final ROC AUC is not a finite number.")

    return float(result)
