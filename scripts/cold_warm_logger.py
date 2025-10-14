#!/usr/bin/env python3
"""
Cold vs Warm Latency Logger (Ollama)

- Forces a true cold-start by restarting the Ollama service before the first call in each series
- Measures TTFT (time to first token) with streaming enabled
- Measures E2E (total) latency
- Logs one JSON line per request to ../data/cold_warm.jsonl
- Uses monotonic clocks for proper timing
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import requests

# ---------- Config (override with CLI flags) ----------
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")        # e.g., mistral, llama3
SERIES = 10                                         # how many cold+warm series to run
WARM_PER_SERIES = 5                                 # warm reps after each cold call
STREAM = True                                       # streaming ON to capture TTFT
MAX_TOKENS = 200                                    # cap output to stabilize E2E variance
TEMPERATURE = 0.2

PROMPTS = {
    "short": "Hi.",
    "medium": "Explain the process of photosynthesis in detail.",
}

OUT_PATH = Path(__file__).parent.parent / "data" / "cold_warm_no_stream.jsonl"
# -----------------------------------------------------


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def restart_ollama():
    """
    Force a cold start by restarting Ollama.
    macOS (default installer) uses launchctl label com.ollama.ollama.
    If your setup differs, use the --restart-cmd flag to provide a custom command.
    """
    try:
        # Kickstart (stop+start) the user agent
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.ollama.ollama"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)  # brief settle window
        return True, None
    except Exception as e:
        return False, str(e)


def run_generate(host: str, model: str, prompt: str, stream: bool, max_tokens: int, temperature: float, timeout=600):
    """
    Call Ollama /api/generate.
    Returns dict: { ttft_ms, latency_ms, output, err_type }
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    start = time.monotonic()
    ttft_ms = None
    output_chunks = []

    try:
        if stream:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    # Decode bytes → str if needed
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    line = raw.strip()

                    if line.startswith("data:"):
                        line = line[5:].strip()

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("done"):
                        break

                    chunk = obj.get("response", "")
                    if chunk and ttft_ms is None:
                        ttft_ms = (time.monotonic() - start) * 1000.0
                    if chunk:
                        output_chunks.append(chunk)

            latency_ms = (time.monotonic() - start) * 1000.0
            return {
                "ttft_ms": round(ttft_ms if ttft_ms is not None else -1, 1),
                "latency_ms": round(latency_ms, 1),
                "output": "".join(output_chunks),
                "err_type": None
            }

        else:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            obj = r.json()
            # Non-streaming: we can’t get TTFT, set -1
            latency_ms = (time.monotonic() - start) * 1000.0
            return {
                "ttft_ms": -1.0,
                "latency_ms": round(latency_ms, 1),
                "output": obj.get("response", ""),
                "err_type": None
            }

    except requests.HTTPError as e:
        return {"ttft_ms": -1.0, "latency_ms": -1.0, "output": "", "err_type": f"HTTPError {e.response.status_code}"}
    except requests.RequestException as e:
        return {"ttft_ms": -1.0, "latency_ms": -1.0, "output": "", "err_type": f"RequestException: {str(e)}"}


def log_jsonl(path: Path, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Cold vs Warm Latency Logger (Ollama)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    ap.add_argument("--model", default=MODEL, help="Model name (ollama)")
    ap.add_argument("--series", type=int, default=SERIES, help="# of cold+warm series")
    ap.add_argument("--warm-per-series", type=int, default=WARM_PER_SERIES, help="# warm requests after each cold")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming (TTFT becomes -1)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--restart-cmd", type=str, default="", help="Custom restart command if launchctl not applicable")
    ap.add_argument("--cooldown-sec", type=float, default=0.5, help="Pause between warm trials")
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS.keys()), help="Subset: short medium")
    ap.add_argument("--out", default=str(OUT_PATH), help="Output JSONL path")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    ensure_parent(out_path)

    # Capture environment provenance once
    provenance = {
        "engine": "ollama",
        "engine_version": subprocess.getoutput("ollama --version").strip(),
        "host_machine": subprocess.getoutput("uname -a").strip()
    }

    for prompt_id in args.prompts:
        if prompt_id not in PROMPTS:
            print(f"[WARN] Unknown prompt key: {prompt_id}. Valid: {list(PROMPTS.keys())}", file=sys.stderr)
            continue
        prompt = PROMPTS[prompt_id]

        for series_idx in range(1, args.series + 1):
            # --- Force cold start
            ok = True
            err = None
            if args.restart_cmd:
                try:
                    subprocess.run(args.restart_cmd, shell=True, check=True)
                    time.sleep(3)
                except Exception as e:
                    ok, err = False, str(e)
            else:
                ok, err = restart_ollama()

            # Cold request (trial_id=1)
            ts = now_iso()
            res = run_generate(
                host=args.host,
                model=args.model,
                prompt=prompt,
                stream=(not args.no_stream),
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            log_jsonl(out_path, {
                "ts": ts,
                "run_id": ts.replace(":", "").replace("-", ""),
                "series_id": series_idx,
                "trial_id": 1,
                "cold_start": True,
                "prompt_id": prompt_id,
                "prompt_chars": len(prompt),
                "ttft_ms": res["ttft_ms"],
                "latency_ms": res["latency_ms"],
                "response_chars": len(res["output"]),
                "model": args.model,
                "stream": not args.no_stream,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "err_type": res["err_type"],
                "restart_ok": ok,
                "restart_err": err,
                **provenance
            })

            # Warm requests (trial_id=2..N)
            for trial_id in range(2, args.warm_per_series + 2):
                time.sleep(args.cooldown_sec)
                ts = now_iso()
                res = run_generate(
                    host=args.host,
                    model=args.model,
                    prompt=prompt,
                    stream=(not args.no_stream),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                log_jsonl(out_path, {
                    "ts": ts,
                    "run_id": ts.replace(":", "").replace("-", ""),
                    "series_id": series_idx,
                    "trial_id": trial_id,
                    "cold_start": False,
                    "prompt_id": prompt_id,
                    "prompt_chars": len(prompt),
                    "ttft_ms": res["ttft_ms"],
                    "latency_ms": res["latency_ms"],
                    "response_chars": len(res["output"]),
                    "model": args.model,
                    "stream": not args.no_stream,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "err_type": res["err_type"],
                    **provenance
                })

    print(f"\n Done. Logs written to: {out_path}")
    print("Next: analyze quantiles (p50/p95), plot cold vs warm TTFT & E2E.")
    

if __name__ == "__main__":
    main()
