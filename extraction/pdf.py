"""
PDF Table Extractor using LiteLLM (supports OpenAI, Anthropic, etc.)
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

from litellm import completion
from litellm.utils import supports_pdf_input


class TableExtractor:
    """Extracts structured data from PDF tables using vision models via LiteLLM"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize extractor

        Args:
            api_key: Optional API key (can also use environment variables)
                     OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
        """
        self.api_key = api_key
        # TODO add guardrails, e.g. max tokens

    def extract(
        self,
        pdf_path: Path,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        fields: List[str] = None,
    ) -> Dict:
        """
        Extract table data from PDF

        Args:
            pdf_path: Path to PDF file
            model: Model name. Examples:
                   - "gpt-4o" (OpenAI)
                   - "claude-3-5-sonnet-20241022" (Anthropic)
                   - "gemini/gemini-1.5-pro" (Google)
            temperature: Sampling temperature
            fields: List of fields to extract

        Returns:
            Dict with 'extractions' key containing list of row dicts
        """
        if fields is None:
            fields = ["costume_description", "highest_price"]

        # Check if model supports PDF input
        if not supports_pdf_input(model):
            raise ValueError(f"Model {model} does not support PDF input")

        # Build prompt
        prompt = self._build_prompt(fields)

        # read pdf as string
        pdf_encoded = self._encode_pdf(pdf_path)

        # Call via LiteLLM with PDF
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Here is a PDF (base64 encoded). Extract the specified fields:\n{pdf_encoded}...",
                },
                # truncated to avoid token blowup
            ],
            api_key=self.api_key,
            temperature=temperature,
        )

        # Parse response
        content = response.choices[0].message.content
        result = self._parse_response(content)

        # Add metadata
        result["model"] = model
        result["temperature"] = temperature

        return result

    def _build_prompt(self, fields: List[str]) -> str:
        """Build extraction prompt"""
        fields_desc = ", ".join(fields)

        prompt = f"""Extract data from the table in this document.

For each row in the table, extract the following fields:
{fields_desc}

Return the data as a JSON object with this exact structure:
{{
  "extractions": [
    {{
      "costume_description": "...",
      "average_price": 123.45
    }},
    ...
  ]
}}

Rules:
- Extract ALL rows from the table
- costume_description should be a string
- highest_price should be a number (no currency symbols)
- Return ONLY valid JSON, no other text
- If handwriting is unclear, make your best interpretation
"""
        return prompt

    def _parse_response(self, content: str) -> Dict:
        """Parse JSON from model response"""

        # try to extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {content}")
            raise ValueError(f"Model returned invalid JSON: {e}")

    def _encode_pdf(self, pdf_path: Path) -> str:
        """Encodes local pdf as base64 string.

        Args:
            pdf_path

        Returns
            pdf as base64 string

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be accessed.
            OSError: For other I/O errors.
        """
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return base64.b64encode(pdf_bytes).decode("utf-8")


def main():
    """Test extractor with multiple models"""
    import os

    # Setup

    extractor = TableExtractor(api_key=os.environ["OPENAI_API_KEY"])
    pdf_path = Path("data/00001.pdf")

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    # Test multiple models
    models = ["gpt-4.1-mini"]

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Testing: {model}")
        print(f"{'=' * 60}")

        try:
            # Check if model supports PDF
            if not supports_pdf_input(model):
                print(f"⚠ Model {model} doesn't support PDF input, skipping")
                continue

            result = extractor.extract(
                pdf_path=pdf_path,
                model=model,
                temperature=0.0,
                fields=["costume_description", "highest_price"],
            )

            print(json.dumps(result, indent=2))
            print(f"✓ Extracted {len(result['extractions'])} rows")

        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()