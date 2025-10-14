import requests
import time
import csv
from datetime import datetime
import os

# Configuration
MODEL = "llama3"  # Change to llama3, phi3, etc.
PROMPTS = [
    "Hi.",
    "Explain the process of photosynthesis.",
    "Write a detailed 4-paragraph essay about the causes of World War I.",
    "Describe a method to simulate an LLM’s latency under constrained memory.",
]

REPEAT = 3  # How many times to repeat each prompt
CSV_FILE = "../data/ollama_benchmark_results.csv"

# Ensure the data folder exists
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Run benchmark
def run_test(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    start = time.time()
    response = requests.post(url, json=payload)
    end = time.time()
    latency = round(end - start, 3)

    if response.status_code == 200:
        output = response.json()["response"]
        return latency, len(prompt), len(output), output
    else:
        return None, len(prompt), 0, f"Error: {response.status_code}"

# Logging
def log_result(row):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "model", "prompt_length", "response_length", "latency_sec", "prompt", "output"])
        writer.writerow(row)

# Run all tests
for prompt in PROMPTS:
    for i in range(REPEAT):
        print(f"Running test: model={MODEL}, prompt={len(prompt)} chars, round={i+1}")
        latency, prompt_len, output_len, output = run_test(prompt)
        log_result([
            datetime.now().isoformat(),
            MODEL,
            prompt_len,
            output_len,
            latency,
            prompt,
            output[:100] + "..."  # Truncate long output for logging
        ])
        time.sleep(1)  # Delay between runs
