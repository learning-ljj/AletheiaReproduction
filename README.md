# AletheiaReproduction

> **An independent Python reproduction of the [Aletheia](https://github.com/google-deepmind/superhuman-reasoning/tree/main/aletheia) framework** by Google DeepMind, built for local experimentation with large language model-based mathematical reasoning.

---

## Background

[Aletheia](https://github.com/google-deepmind/superhuman-reasoning) is a framework introduced in Google DeepMind's *Superhuman Reasoning* research. It solves competition-level mathematics problems through an iterative refinement loop:

```
Generator => Verifier (3-phase) => Reviser / Generator => ...
```

This repository reimplements the core pipeline in Python with an OpenAI-compatible client, targeting the DeepSeek V3.2 model (via DeepSeek official API or Volcano Engine).

---

## Architecture

```
AletheiaReproduction/
+-- main.py                     # CLI entry point
+-- config/
|   +-- settings.yaml           # LLM provider / model config (env-var based)
|   +-- prompts.yaml            # System & user prompts for Generator / Verifier / Reviser
+-- src/
|   +-- core/
|   |   +-- agent.py            # AletheiaAgent + call_generator / verifier / reviser
|   |   +-- state.py            # ProofState, VerificationLog, VerificationDecision
|   |   +-- config.py           # load_config(), load_prompts()
|   +-- models/
|   |   +-- llm_client.py       # LLMClient (streaming + tool-calling, thinking protocol)
|   +-- tools/
|   |   +-- registry.py         # Tool JSON schemas + execute_tool() dispatcher
|   |   +-- code_executor.py    # run_python - subprocess sandbox (sympy / numpy)
|   |   +-- web_search.py       # search_arxiv, read_arxiv_latex (arXiv API)
|   |   +-- wiki_search.py      # search_wikipedia (MediaWiki API + HTML cleanup)
|   +-- utils/
|       +-- logger.py           # JSONL logging + human-readable Markdown export
|       +-- parser.py           # Solution & verdict extraction from LLM output
|       +-- evaluator.py        # Answer normalisation & equality check
|       +-- data_loader.py      # IMO Bench CSV loaders
+-- scripts/
|   +-- run_imobench.py         # Batch benchmark runner
+-- data/
    +-- imobench/               # IMO Bench CSV datasets (download separately)
```

### Agent Loop

```
Problem Text
    |
    v
[GENERATOR]  -> thinking + preliminary solution
    |
    v
[VERIFIER]   Phase 1 - read & plan
             Phase 2 - tool calls (run_python / wiki / arxiv)
             Phase 3 - verdict [DECISION]
    |
+---+--------------------+-------------------+
| CORRECT                | MINOR_FLAW        | CRITICAL_FLAW
| -> return answer       | -> [REVISER]      | -> [GENERATOR]
+------------------------+-------------------+
```

| Decision | Meaning | Next step |
|---|---|---|
| `CORRECT` | No logical errors found | Terminate, return answer |
| `MINOR_FLAW` | Justification gap or minor error | Route to Reviser |
| `CRITICAL_FLAW` | Fundamental logical error | Route back to Generator |

---

## Setup

### Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 1. Clone & Install

```powershell
git clone https://github.com/learning-ljj/AletheiaReproduction.git
cd AletheiaReproduction

# Recommended: uv
uv pip install -r requirements.txt

# Alternative: pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```powershell
copy .env.example .env
# Edit .env and fill in your real API keys
```

`.env` key fields:

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `deepseek` or `volcano` |
| `DEEPSEEK_API_KEY` | if deepseek | DeepSeek platform API key |
| `VOLCANO_API_KEY` | if volcano | Volcano Engine API key |
| `VOLCANO_BASE_URL` | if volcano | e.g. `https://ark.cn-beijing.volces.com/api/v3` |
| `VOLCANO_MODEL` | if volcano | e.g. `deepseek-v3-2-251201` |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Avoids S2 rate-limiting |

---

## Usage

### Single Problem (CLI)

```powershell
# From a file
python main.py data/test_problems/proofbench_sample.txt --max-turns 3 --log data/logs/out.md

# Inline text
python main.py --problem "Prove that for all n>=1, n^2+n is even." --max-turns 1
```

### Batch Benchmark (IMO Bench)

Download datasets from [IMO Bench](https://imobench.github.io) and place CSV files in `data/imobench/`.

```powershell
python scripts/run_imobench.py --dataset answerbench --count 10 --max-turns 3
```

---

## Tools

| Tool | Function |
|---|---|
| `run_python` | Execute Python code in a subprocess sandbox (sympy, numpy, scipy available) |
| `search_wikipedia` | Query Wikipedia and return cleaned plain text |
| `search_arxiv` | Query arXiv API and return paper titles + abstracts |
| `read_arxiv_latex` | Download and extract arXiv LaTeX source |

---

## Evaluation Datasets

Datasets from [IMO Bench](https://imobench.github.io) - download separately:

| Dataset | File | Description |
|---|---|---|
| AnswerBench | `data/imobench/answerbench.csv` | Short-answer competition problems |
| ProofBench | `data/imobench/proofbench.csv` | Proof-based problems |
| GradingBench | `data/imobench/gradingbench.csv` | Human-graded solution pairs |

---

## License & Attribution

This project is a **derivative work** based on [Aletheia](https://github.com/google-deepmind/superhuman-reasoning/tree/main/aletheia) by Google DeepMind, licensed under the **Apache License 2.0**.

- **Original work**: Copyright (c) Google LLC - [superhuman-reasoning](https://github.com/google-deepmind/superhuman-reasoning)
- **This reproduction**: Copyright (c) 2025 learning-ljj

Key differences from the original:

- Implemented in Python with an OpenAI-compatible API client
- Supports DeepSeek V3.2 interleaved thinking (reasoning_content + content)
- Extended tool registry: Wikipedia and arXiv integrations added
- JSONL structured logging with Markdown human-readable export
- CLI (main.py) and batch benchmark runner (run_imobench.py)

See [LICENSE](LICENSE) for full terms.

---

## Disclaimer

This is an **independent reproduction** created for research and personal learning. It is **not** an official Google or Google DeepMind product.