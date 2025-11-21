# FUTURE-AI Metrics LLM

This repository contains LLM-based evaluation tools for assessing AI clinical universality metrics in medical imaging research papers.

## Overview

The project provides two implementations for evaluating research papers against FUTURE-AI universality metrics:

1. **OpenAI-based evaluation** (`universality_agent_chatgpt.py`) - Uses OpenAI's GPT-4o-mini via LangChain
2. **Ollama-based evaluation** (`universality_agent_ollama.py`) - Uses local Ollama with LLaMA 3.1

## Features

- PDF document parsing and text extraction
- Keyword-based evidence retrieval from research papers
- LLM-powered metric evaluation with structured JSON output
- Support for multiple evaluation criteria and metrics
- Configurable LLM providers (OpenAI or Ollama)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/xrafael/FUTURE-AI-METRICS-LLM.git
cd FUTURE-AI-METRICS-LLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For OpenAI version, set your API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. For Ollama version, ensure Ollama is installed and running with LLaMA 3.1:
```bash
ollama pull llama3.1
```

## Usage

### OpenAI Version

```bash
python universality_agent_chatgpt.py
```

### Ollama Version

```bash
python universality_agent_ollama.py
```

## Configuration

- **Metrics file**: `future_ai_metrics.json` - Contains the evaluation criteria and metrics
- **Paper path**: Update the `paper_pdf_path` variable in the script to point to your PDF file
- **Output**: Results are saved to `results/universality_report.json`

## Project Structure

```
.
├── universality_agent_chatgpt.py    # OpenAI-based evaluator
├── universality_agent_ollama.py     # Ollama-based evaluator
├── future_ai_metrics.json          # Metrics configuration
├── papers/                          # PDF papers directory (gitignored)
└── results/                         # Output directory (gitignored)
```

## Evaluation Criteria

The tool evaluates papers against three main criteria:

1. **Operational Applicability**: Computational resources, scanner/software compatibility, medical sites, and countries
2. **Interoperability Standards**: Model formats, image formats, communications, and clinical terms
3. **Clinical Validation**: Diverse demographics, local populations, external data, and multiple clinical sites

## Notes

- The current implementation uses simple keyword-based search for evidence retrieval
- Future improvements could include semantic search via vector embeddings
- LLM calls can be parallelized for better performance on large PDFs

## License

[Add your license here]

