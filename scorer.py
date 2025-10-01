from typing import Callable, Dict, List, Optional

from pydantic import BaseModel

# ----------------------
# Data models
# ----------------------


class Line(BaseModel):
    """Represents one line item in a PDF extraction."""

    line_id: int
    average_price: Optional[float] = None


class GroundTruthEntry(BaseModel):
    """Ground truth for one PDF file."""

    lines: List[Line]  # List of line items for this PDF


class PredictionEntry(BaseModel):
    """Extraction result for one PDF file."""

    extractions: List[Line]  # List of predicted lines


class FieldScore(BaseModel):
    """Score for a single field aggregated across all PDFs."""

    field_name: str
    metric: Optional[float]  # Metric value (e.g., MAE, RMSE)


class ScoreResult(BaseModel):
    """Aggregated scoring result across multiple PDFs."""

    total_lines: int  # Total number of ground truth lines across all PDFs
    matched_lines: int  # Number of lines matched by line_id across all PDFs
    coverage: float  # Fraction of lines matched across all PDFs
    field_scores: List[FieldScore]  # Aggregate metric for each field


class MultiPDFScorer:
    """Scores multiple PDFs and computes aggregate metrics across all lines."""

    def __init__(
        self, field_metrics: Dict[str, Callable[[List[float], List[float]], float]]
    ):
        """
        Args:
            field_metrics: Mapping from field name to metric function.
                           Each function should accept (y_true, y_pred) lists and return a numeric score.
        """
        self.field_metrics = field_metrics

    def score(
        self, ground_truth: List[GroundTruthEntry], predictions: List[PredictionEntry]
    ) -> ScoreResult:
        """
        Compute coverage and aggregate per-field metrics across all PDFs.

        Args:
            ground_truth: List of ground truth entries.
            predictions: List of extraction entries.

        Returns:
            ScoreResult containing aggregated coverage and per-field metrics.
        """
        if len(ground_truth) != len(predictions):
            raise ValueError(
                "Number of ground truth entries and predictions must match"
            )

        total_lines = 0  # Count of all ground truth lines
        matched_lines = 0  # Count of lines with a matching line_id

        # collect all true/predicted values for each field across all PDFs
        field_values: Dict[str, List[float]] = {f: [] for f in self.field_metrics}
        field_pred_values: Dict[str, List[float]] = {f: [] for f in self.field_metrics}

        # loop over each PDF
        for gt_entry, pred_entry in zip(ground_truth, predictions):

            # map line_id to line for fast lookup
            gt_lines = {l.line_id: l for l in gt_entry.lines}
            pred_lines = {l.line_id: l for l in pred_entry.extractions}

            # update total and matched line counts
            total_lines += len(gt_lines)
            matched_lines += sum(1 for lid in gt_lines if lid in pred_lines)

            # aggregate field values across all PDFs
            for field_name in self.field_metrics:
                for lid, gt_line in gt_lines.items():
                    if lid in pred_lines:
                        gt_val = getattr(gt_line, field_name, None)
                        pred_val = getattr(pred_lines[lid], field_name, None)

                        # only include lines where both GT and prediction have a value
                        if gt_val is not None and pred_val is not None:
                            field_values[field_name].append(gt_val)
                            field_pred_values[field_name].append(pred_val)

        # compute overall coverage
        coverage = matched_lines / total_lines if total_lines else 0.0

        # compute aggregate metric for each field
        field_scores = []
        for field_name, metric_fn in self.field_metrics.items():
            y_true = field_values[field_name]
            y_pred = field_pred_values[field_name]
            score = metric_fn(y_true, y_pred) if y_true else None
            field_scores.append(FieldScore(field_name=field_name, metric=score))

        # return a single aggregated score result
        return ScoreResult(
            total_lines=total_lines,
            matched_lines=matched_lines,
            coverage=coverage,
            field_scores=field_scores,
        )


# metrics
def mae(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error."""
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Squared Error."""
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


if __name__ == "__main__":
    import json

    # Load ground truth JSON
    with open("data/ground_truth.json") as f:
        gt_data = [GroundTruthEntry(**entry) for entry in json.load(f)]

    # Example predictions for multiple PDFs
    predictions = [
        PredictionEntry(
            extractions=[
                Line(line_id=1, average_price=41.45),
                Line(line_id=2, average_price=67.00),
                Line(line_id=3, average_price=63.47),
            ]
        ),
    ]

    # Initialize scorer with multiple fields and metrics
    scorer = MultiPDFScorer(field_metrics={"average_price": mae})

    # Compute aggregated scores
    result = scorer.score(gt_data, predictions)

    # Print results as JSON
    print(result.model_dump_json(indent=2))