import os
import csv
import time
from datetime import datetime
import requests
from pathlib import Path

# ====== CONFIG ======
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELS = ["mistral"]          # e.g., ["mistral", "llama3", "phi3"]
REPEAT = 3                    # runs per prompt
PROMPTS = [
    "Hi.",
    "Explain the process of photosynthesis.",
    "Write a detailed 4-paragraph essay about the causes of World War I.",
    "Describe a method to simulate an LLM’s latency under constrained memory.",
]
CSV_FILE = Path(__file__).parent.parent / "data" / "ollama_benchmark_results.csv"
# ====================

CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

def call_ollama(model: str, prompt: str):
    """Call Ollama's /api/generate; return (latency_sec, output_text) or raise."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    start = time.time()
    r = requests.post(url, json=payload, timeout=600)
    latency = round(time.time() - start, 3)

    r.raise_for_status()
    out = r.json().get("response", "")
    return latency, out

def log_row(row: list):
    """Append a CSV row; create header if file does not exist yet."""
    new_file = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "timestamp", "model", "prompt_length", "response_length",
                "latency_sec", "cold_start", "prompt", "output_preview"
            ])
        w.writerow(row)

def main():
    warmed = set()  # track which models have received at least one call

    for model in MODELS:
        for prompt in PROMPTS:
            for trial in range(1, REPEAT + 1):
                cold_start = model not in warmed
                print(f"[{datetime.now().isoformat(timespec='seconds')}] "
                      f"model={model} cold_start={cold_start} prompt_len={len(prompt)} trial={trial}")

                try:
                    latency, output = call_ollama(model, prompt)
                    warmed.add(model)  # after first successful call, it's warm
                    log_row([
                        datetime.now().isoformat(),
                        model,
                        len(prompt),
                        len(output),
                        latency,
                        cold_start,
                        prompt,
                        (output[:120] + "…") if len(output) > 120 else output
                    ])
                except requests.HTTPError as e:
                    log_row([
                        datetime.now().isoformat(),
                        model,
                        len(prompt),
                        0,
                        -1,  # signal error
                        cold_start,
                        prompt,
                        f"HTTPError {e.response.status_code}: {getattr(e.response, 'text', '')[:120]}"
                    ])
                except requests.RequestException as e:
                    log_row([
                        datetime.now().isoformat(),
                        model,
                        len(prompt),
                        0,
                        -1,
                        cold_start,
                        prompt,
                        f"RequestException: {str(e)[:120]}"
                    ])

                time.sleep(1)  # small gap between trials

if __name__ == "__main__":
    main()
