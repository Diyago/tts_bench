"""Post-process ASR output with LLM to fix errors and recompute WER/CER.

Uses the same normalization as evaluate.py for consistent metrics.
Updates benchmark_report.xlsx with wer_llm / cer_llm columns.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import logging
import time
import requests
import pandas as pd
from jiwer import cer, wer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate import normalize_for_eval
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("llm_postprocess")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3-chat:latest"

SYSTEM_PROMPT = """Ты — пост-процессор текста автоматического распознавания речи (ASR).
Тебе дан текст, распознанный из русской речи. В нём могут быть ошибки:
- Опечатки в словах
- Неправильная транслитерация английских терминов (например "обзирвабилити" → "observability")
- Лишние/пропущенные слова
- Неправильные окончания

Исправь ошибки, сохранив оригинал как можно ближе.
Не добавляй слова, которых нет. Не убирай слова, которые есть.
Исправляй только очевидные ошибки распознавания.
Если слово может быть правильным — не трогай его.

Выведи ТОЛЬКО исправленный текст, без пояснений."""


def postprocess_text(text: str) -> str:
    """Send text to LLM for post-processing."""
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return text  # fallback to original


def main():
    df = pd.read_csv("results/per_sample_results.csv")

    # Process all models
    models = df["model_name"].unique()
    print(f"Models to process: {', '.join(models)}\n")

    all_results = []
    model_summary = {}

    for model_name in models:
        model_df = df[df["model_name"] == model_name].copy()
        model_df = model_df.drop_duplicates(subset="audio_filename", keep="last")

        print(f"=== {model_name} ({len(model_df)} samples) ===")

        refs_raw = []
        hyps_raw = []
        hyps_fixed = []
        filenames = []
        improved = 0
        degraded = 0
        same = 0

        for i, row in model_df.iterrows():
            ref = str(row["reference_original"])
            hyp = str(row["hypothesis_raw"])

            # LLM post-processing
            hyp_fixed = postprocess_text(hyp)

            refs_raw.append(ref)
            hyps_raw.append(hyp)
            hyps_fixed.append(hyp_fixed)
            filenames.append(row["audio_filename"])

            # Per-sample comparison using consistent normalization
            ref_norm = normalize_for_eval(ref)
            hyp_norm = normalize_for_eval(hyp)
            fixed_norm = normalize_for_eval(hyp_fixed)

            if ref_norm:
                try:
                    w_before = wer([ref_norm], [hyp_norm])
                    w_after = wer([ref_norm], [fixed_norm])
                except Exception:
                    w_before, w_after = 1.0, 1.0
            else:
                w_before, w_after = 0.0, 0.0

            if w_after < w_before:
                improved += 1
            elif w_after > w_before:
                degraded += 1
            else:
                same += 1

            all_results.append({
                "model_name": model_name,
                "audio_filename": row["audio_filename"],
                "reference_original": ref,
                "hypothesis_original": hyp,
                "hypothesis_fixed": hyp_fixed,
                "wer_before": round(w_before, 4),
                "wer_after": round(w_after, 4),
            })

            if (i + 1) % 10 == 0 or i == model_df.index[-1]:
                print(f"  [{len(refs_raw)}/{len(model_df)}] processed")

            time.sleep(0.3)

        # Corpus-level WER/CER using consistent normalization
        refs_norm = [normalize_for_eval(r) for r in refs_raw]
        hyps_orig_norm = [normalize_for_eval(h) for h in hyps_raw]
        hyps_fixed_norm = [normalize_for_eval(h) for h in hyps_fixed]

        # Filter empty
        valid = [(r, ho, hf) for r, ho, hf in zip(refs_norm, hyps_orig_norm, hyps_fixed_norm) if r.strip()]
        if valid:
            rv, hov, hfv = zip(*valid)
            rv, hov, hfv = list(rv), list(hov), list(hfv)
        else:
            rv, hov, hfv = [], [], []

        if rv:
            wer_before = wer(rv, hov)
            cer_before = cer(rv, hov)
            wer_after = wer(rv, hfv)
            cer_after = cer(rv, hfv)
        else:
            wer_before = cer_before = wer_after = cer_after = 0.0

        model_summary[model_name] = {
            "wer_llm": round(wer_after, 4),
            "cer_llm": round(cer_after, 4),
            "wer_before": round(wer_before, 4),
            "cer_before": round(cer_before, 4),
            "improved": improved,
            "degraded": degraded,
            "same": same,
        }

        delta = (wer_before - wer_after) / max(wer_before, 1e-9) * 100
        print(f"  RESULT: WER {wer_before:.4f} → {wer_after:.4f} ({delta:+.1f}%)")
        print(f"          CER {cer_before:.4f} → {cer_after:.4f}")
        print(f"          Improved: {improved}, Degraded: {degraded}, Same: {same}")
        print()

    # Save detailed per-sample results
    out_df = pd.DataFrame(all_results)
    out_df.to_csv("results/llm_postprocess_results.csv", index=False)
    print("Saved to results/llm_postprocess_results.csv")

    # Update benchmark_report.xlsx
    update_excel_report(model_summary)

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Model':<35} {'WER':>8} {'WER LLM':>8} {'Δ%':>7} {'CER':>8} {'CER LLM':>8}")
    print("-" * 80)
    for model_name in models:
        s = model_summary[model_name]
        delta = (s["wer_before"] - s["wer_llm"]) / max(s["wer_before"], 1e-9) * 100
        print(f"{model_name:<35} {s['wer_before']:>8.4f} {s['wer_llm']:>8.4f} {delta:>+6.1f}% {s['cer_before']:>8.4f} {s['cer_llm']:>8.4f}")


def update_excel_report(model_summary: dict):
    """Add wer_llm and cer_llm columns to Summary and Per-model Stats sheets."""
    excel_path = Path("results/benchmark_report.xlsx")
    if not excel_path.exists():
        print(f"Warning: {excel_path} not found, skipping Excel update")
        return

    print(f"\nUpdating {excel_path} with LLM post-processing results...")

    # Read all sheets
    xls = pd.ExcelFile(excel_path)
    sheets = {}
    for name in xls.sheet_names:
        sheets[name] = pd.read_excel(excel_path, sheet_name=name)

    # Update Summary sheet
    if "Summary" in sheets:
        df_summary = sheets["Summary"]
        # Deduplicate: keep last row per model
        df_summary = df_summary.drop_duplicates(subset="model_name", keep="last")
        wer_llm_col = []
        cer_llm_col = []
        wer_base_col = []
        cer_base_col = []
        for _, row in df_summary.iterrows():
            name = row["model_name"]
            if name in model_summary:
                wer_llm_col.append(model_summary[name]["wer_llm"])
                cer_llm_col.append(model_summary[name]["cer_llm"])
                wer_base_col.append(model_summary[name]["wer_before"])
                cer_base_col.append(model_summary[name]["cer_before"])
            else:
                wer_llm_col.append(row.get("wer"))
                cer_llm_col.append(row.get("cer"))
                wer_base_col.append(row.get("wer"))
                cer_base_col.append(row.get("cer"))
        # Replace wer/cer with baseline from same data used for LLM post-processing
        df_summary["wer"] = wer_base_col
        df_summary["cer"] = cer_base_col
        df_summary["wer_llm"] = wer_llm_col
        df_summary["cer_llm"] = cer_llm_col
        sheets["Summary"] = df_summary

    # Update Per-model Stats sheet
    if "Per-model Stats" in sheets:
        df_stats = sheets["Per-model Stats"]
        wer_llm_col = []
        cer_llm_col = []
        for _, row in df_stats.iterrows():
            name = row["model_name"]
            if name in model_summary:
                wer_llm_col.append(model_summary[name]["wer_llm"])
                cer_llm_col.append(model_summary[name]["cer_llm"])
            else:
                wer_llm_col.append(None)
                cer_llm_col.append(None)
        df_stats["wer_llm"] = wer_llm_col
        df_stats["cer_llm"] = cer_llm_col
        sheets["Per-model Stats"] = df_stats

    # Add LLM Post-processing summary sheet
    llm_rows = []
    for model_name, s in model_summary.items():
        delta = (s["wer_before"] - s["wer_llm"]) / max(s["wer_before"], 1e-9) * 100
        llm_rows.append({
            "model_name": model_name,
            "wer_original": s["wer_before"],
            "cer_original": s["cer_before"],
            "wer_llm": s["wer_llm"],
            "cer_llm": s["cer_llm"],
            "wer_delta_%": round(delta, 1),
            "samples_improved": s["improved"],
            "samples_degraded": s["degraded"],
            "samples_same": s["same"],
        })
    sheets["LLM Post-process"] = pd.DataFrame(llm_rows)

    # Add "bench results" sheet: merge Details + LLM post-processing
    if "Details" in sheets:
        df_details = sheets["Details"].copy()
        df_llm = pd.read_csv("results/llm_postprocess_results.csv")
        # Deduplicate LLM results (one row per model+audio_filename)
        df_llm = df_llm.drop_duplicates(subset=["model_name", "audio_filename"], keep="last")
        # Merge on model_name + audio_filename
        df_bench = df_details.merge(
            df_llm[["model_name", "audio_filename", "hypothesis_fixed", "wer_before", "wer_after"]],
            on=["model_name", "audio_filename"],
            how="left",
        )
        # Compute CER for LLM-fixed hypothesis
        from scripts.evaluate import normalize_for_eval
        from jiwer import cer as jiwer_cer
        cer_llm_list = []
        for _, row in df_bench.iterrows():
            ref_n = normalize_for_eval(str(row.get("reference_original", "")))
            hyp_n = normalize_for_eval(str(row.get("hypothesis_fixed", "")))
            if ref_n and hyp_n:
                try:
                    cer_llm_list.append(round(jiwer_cer([ref_n], [hyp_n]), 4))
                except Exception:
                    cer_llm_list.append(None)
            else:
                cer_llm_list.append(None)
        df_bench["cer_llm"] = cer_llm_list
        # Rename wer_before/after for clarity
        df_bench = df_bench.rename(columns={"wer_before": "wer_orig", "wer_after": "wer_llm"})
        # Reorder columns: text fields, then all metrics
        bench_cols = [
            "model_name", "augmentation", "audio_filename", "duration_sec", "duration_bucket",
            "reference_original", "reference_normalized",
            "hypothesis_raw", "hypothesis_normalized", "hypothesis_fixed",
            "is_correct", "wer", "cer", "wer_llm", "cer_llm", "wer_orig",
            "rtf", "inference_time_sec", "gpu_memory_peak_mb",
            "audio_path", "source_audio_path", "source_audio_filename",
        ]
        existing_bench = [c for c in bench_cols if c in df_bench.columns]
        sheets["bench results"] = df_bench[existing_bench]

    # Write back
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for col_idx, col in enumerate(df_sheet.columns, 1):
                max_len = max(
                    len(str(col)),
                    df_sheet[col].astype(str).str.len().max() if len(df_sheet) > 0 else 0,
                )
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 3, 50)

    print(f"Updated {excel_path}")


if __name__ == "__main__":
    main()
