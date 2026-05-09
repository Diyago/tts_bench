"""Evaluate semantic preservation of ASR output using a local LLM via Ollama."""

import json
import sys
import time
import requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3-chat:latest"

SYSTEM_PROMPT = """Ты — оценщик качества автоматического распознавания речи (ASR).
Тебе даны два текста:
1. Эталонная транскрипция (reference) — что было сказано на самом деле.
2. Гипотеза ASR (hypothesis) — что распознала модель.

Твоя задача — оценить, сохранился ли СМЫСЛ высказывания, несмотря на ошибки распознавания.

Отвечай СТРОГО в JSON формате без markdown:
{
  "meaning_preserved": true/false,
  "severity": "none" | "minor" | "moderate" | "severe",
  "explanation": "краткое объяснение на русском"
}

Правила:
- "none" — ошибок нет или они косметические (ё/е, пунктуация)
- "minor" — мелкие искажения, смысл понятен (опечатки в словах, пропуск союзов)
- "moderate" — заметные искажения, но основной смысл угадывается
- "severe" — смысл потерян или сильно искажён, непонятно о чём речь
"""


def evaluate_sample(reference: str, hypothesis: str) -> dict:
    """Send a single pair to Ollama and get judgment."""
    prompt = f"Эталон: {reference}\n\nГипотеза: {hypothesis}"
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["response"].strip()
        # Try to extract JSON from response
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            return json.loads(json_str)
        return {"meaning_preserved": False, "severity": "unknown", "explanation": text}
    except Exception as e:
        return {"meaning_preserved": False, "severity": "error", "explanation": str(e)}


def main():
    df = pd.read_csv("results/per_sample_results.csv")
    whisper = df[df["model_name"] == "whisper-large-v3_beam5"].copy()
    whisper = whisper.drop_duplicates(subset="audio_filename", keep="first")

    # Only evaluate samples with errors
    errors = whisper[whisper["wer"] > 0].sort_values("wer", ascending=False).reset_index(drop=True)
    print(f"Evaluating {len(errors)} samples with errors...\n")

    results = []
    for i, row in errors.iterrows():
        ref = str(row["reference_normalized"])
        hyp = str(row["hypothesis_normalized"])
        filename = row["audio_filename"]

        print(f"[{i+1}/{len(errors)}] {filename} (WER={row['wer']:.3f})...", end=" ", flush=True)
        judgment = evaluate_sample(ref, hyp)
        print(f"{judgment['severity']} — preserved={judgment['meaning_preserved']}")

        results.append({
            "audio_filename": filename,
            "wer": row["wer"],
            "cer": row["cer"],
            "duration_sec": row["duration_sec"],
            "reference": ref,
            "hypothesis": hyp,
            "meaning_preserved": judgment.get("meaning_preserved", False),
            "severity": judgment.get("severity", "unknown"),
            "explanation": judgment.get("explanation", ""),
        })
        time.sleep(0.5)  # small delay between requests

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv("results/semantic_eval.csv", index=False)
    print(f"\nSaved to results/semantic_eval.csv")

    # Summary
    print("\n=== SEMANTIC EVALUATION SUMMARY ===")
    total = len(results)
    preserved = sum(1 for r in results if r["meaning_preserved"])
    print(f"Total evaluated: {total}")
    print(f"Meaning preserved: {preserved}/{total} ({preserved/total*100:.1f}%)")
    print(f"Meaning lost: {total - preserved}/{total} ({(total-preserved)/total*100:.1f}%)")

    print("\nBy severity:")
    from collections import Counter
    sev_counts = Counter(r["severity"] for r in results)
    for sev in ["none", "minor", "moderate", "severe", "unknown", "error"]:
        if sev in sev_counts:
            print(f"  {sev}: {sev_counts[sev]}")

    print("\nSamples where meaning is LOST:")
    for r in results:
        if not r["meaning_preserved"]:
            print(f"  - {r['audio_filename']} (WER={r['wer']:.3f}, {r['severity']})")
            print(f"    {r['explanation']}")


if __name__ == "__main__":
    main()
