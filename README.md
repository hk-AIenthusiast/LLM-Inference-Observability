# LLM Inference Observability Project

This project benchmarks and analyzes latency, token usage, and cost tradeoffs when calling large language models (LLMs) under various conditions. The goal is to simulate how LLM-based systems behave in production and propose infrastructure-level optimizations to improve inference performance.

## 📌 Goals
- Measure latency, length scaling, stress test results across different prompt sizes, models, and streaming settings.
- Simulate tool use and assess its impact on response time.
- Propose product and infra recommendations to reduce latency and cost.

## 📁 Structure
- `notebooks/`: Exploratory experiments in Jupyter/Colab
- `scripts/`: Reproducible Python scripts for benchmarking
- `data/`: Raw CSV logs of tests
- `charts/`: Visualizations of trends
- `results/`: Summaries, aggregated metrics

## 🚀 To Run
1. Install dependencies:
   ```bash
   pip install openai pandas matplotlib
   ```
2. Add your OpenAI API key.
3. Run scripts in `/scripts` or use `/notebooks` for interactive exploration.

## 🧠 Author
Harika Kolli – TPM focused on inference systems and AGI infrastructure.
