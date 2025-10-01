import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from litellm import completion
from litellm.utils import supports_pdf_input
from tabulate import tabulate  # for pretty table output

from extraction.pdf import TableExtractor
from scorer import (  # your scorer module
    GroundTruthEntry,
    MultiPDFScorer,
    PredictionEntry,
    mae,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Load experiment config from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_ground_truth(path: Path) -> List[GroundTruthEntry]:
    """Load ground truth JSON into Pydantic objects."""
    with open(path, "r") as f:
        data = json.load(f)
    return [GroundTruthEntry(**entry) for entry in data]


def main():

    # load experiment
    config = load_config(Path("configs/experiment.yaml"))
    pdf_files = config["pdf_files"]  # list of local PDF paths
    models = config["models"]  # list of model names
    temperature = config.get("temperature", 0.0)

    # load ground truth
    ground_truth = load_ground_truth(Path(config["ground_truth_json"]))
    logger.info("loaded ground truth")

    # initialise scorer (coverage + MAE for average_price)
    scorer = MultiPDFScorer(field_metrics={"average_price": mae})
    logger.info("initialised pdf scorer")

    # initialise pdf extractor
    extractor = TableExtractor(api_key=os.environ["OPENAI_API_KEY"])
    logger.info("initialised pdf parser")

    # store results for table
    table_rows = []

    # run extraction
    for model_name in models:
        predictions = []
        for pdf_path in pdf_files:

            # run extraction

            # check if model supports PDF
            if not supports_pdf_input(model_name):
                logger.info(f"⚠ Model {model_name} doesn't support PDF input, skipping")
                continue

            pdf_extraction = extractor.extract(
                pdf_path=pdf_path,
                model=model_name,
                temperature=temperature,
                fields=["average_price"],
            )

            logger.info(json.dumps(pdf_extraction, indent=2))
            logger.info(f"✓ Extracted {len(pdf_extraction['extractions'])} rows")

            predictions.append(PredictionEntry(**pdf_extraction))

        # Score aggregate metrics across all PDFs
        result = scorer.score(ground_truth, predictions)

        # Append model results
        coverage = result.coverage
        mae_val = next(
            (
                fs.metric
                for fs in result.field_scores
                if fs.field_name == "average_price"
            ),
            None,
        )
        table_rows.append([model_name, f"{coverage:.2%}", f"{mae_val:.2f}"])

    # ----------------------
    # Print results table
    # ----------------------
    headers = ["Model", "Coverage", "MAE"]
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()