# 🏟️ PDF Extractor Arena

A service for testing LLM-based PDF extraction pipelines against other.

## Features

* LLM-based pdf parsing via `litellm`
* Supports all common model providers OpenAI, Anthropic
* you can configure experiment parameters via `yaml`
* supports custom metrics per field of interest

## Future work

1. Turn into API service via `fastapi`
2. Stronger error handling & retry mechanism
3. Guardrails
   * cost control