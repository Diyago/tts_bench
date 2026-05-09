"""
Evaluation script for ASR benchmark.

Computes:
  - WER (Word Error Rate)
  - CER (Character Error Rate)
  - RTF (Real-Time Factor)
  - Latency per sample
  - GPU memory peak
  - Error breakdown (substitutions, deletions, insertions)
  - WER by segment length bucket (short/medium/long)

Exports:
  - CSV summary + per-sample
  - Excel (.xlsx) with detailed sheets:
    - Summary - aggregated metrics per model
    - Details - per-sample: reference text, hypothesis text, WER, etc.
    - Errors - worst predictions for analysis

Usage:
    python scripts/evaluate.py
    (or called from run_benchmark.py)
"""

import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jiwer import cer, process_words, wer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("evaluate")


# --- Text normalization ---------------------------------------
def normalize_for_eval(text: str) -> str:
    """
    Normalize predicted text for fair comparison with reference.
    Keeps both Russian and English letters, digits, spaces.
    Applies ё→е normalisation when config.NORMALIZE_YO_YE is True.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    if config.NORMALIZE_YO_YE:
        text = text.replace("ё", "е")
    # Remove punctuation but keep letters/digits/spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Keep only allowed chars (Russian + English + digits + space)
    allowed = set(ch.replace("ё", "е") if config.NORMALIZE_YO_YE else ch
                  for ch in config.ALLOWED_CHARS)
    text = "".join(ch for ch in text if ch in allowed)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Duration buckets -----------------------------------------
def get_duration_bucket(duration_sec: float) -> str:
    """Classify audio duration into short/medium/long."""
    if duration_sec < 5:
        return "short"
    elif duration_sec < 15:
        return "medium"
    else:
        return "long"


# --- Core evaluation ------------------------------------------
def compute_metrics(
    references: list[str],
    hypotheses: list[str],
    durations: list[float],
    inference_times: list[float],
    gpu_memory_peaks: list[float],
    model_name: str = "",
) -> dict:
    """
    Compute all benchmark metrics.

    Returns dict with WER, CER, RTF, latency, GPU mem, error types, etc.
    """
    # Normalize
    refs_norm = [normalize_for_eval(r) for r in references]
    hyps_norm = [normalize_for_eval(h) for h in hypotheses]

    # Filter out empty pairs
    valid = [(r, h, d, t, g) for r, h, d, t, g in
             zip(refs_norm, hyps_norm, durations, inference_times, gpu_memory_peaks)
             if r.strip()]

    if not valid:
        logger.error("No valid reference-hypothesis pairs!")
        return {}

    refs, hyps, durs, times, gpus = zip(*valid)
    refs, hyps = list(refs), list(hyps)
    durs, times, gpus = list(durs), list(times), list(gpus)

    # --- Global WER & CER ---------------------------------
    overall_wer = wer(refs, hyps)
    overall_cer = cer(refs, hyps)

    # --- Error breakdown ----------------------------------
    wo = process_words(refs, hyps)
    total_words = wo.hits + wo.substitutions + wo.deletions
    sub_rate = wo.substitutions / max(total_words, 1)
    del_rate = wo.deletions / max(total_words, 1)
    ins_rate = wo.insertions / max(total_words, 1)

    # --- Timing metrics ----------------------------------
    total_audio_sec = sum(durs)
    total_inference_sec = sum(times)
    rtf = total_inference_sec / max(total_audio_sec, 0.001)
    avg_latency = np.mean(times)
    p50_latency = np.percentile(times, 50)
    p95_latency = np.percentile(times, 95)

    # --- GPU memory ---------------------------------------
    max_gpu_mem = max(gpus) if gpus else 0.0
    avg_gpu_mem = np.mean(gpus) if gpus else 0.0

    # --- Per-sample WER (mean, NOT corpus-level) ----------
    sample_wers = []
    for r, h in zip(refs, hyps):
        try:
            sw = wer([r], [h])
        except Exception:
            sw = 1.0
        sample_wers.append(sw)

    wer_mean_per_sample = float(np.mean(sample_wers))

    # --- WER by duration bucket ---------------------------
    bucket_wers = {"short": [], "medium": [], "long": []}
    for r, h, d in zip(refs, hyps, durs):
        bucket = get_duration_bucket(d)
        try:
            sw = wer([r], [h])
        except Exception:
            sw = 1.0
        bucket_wers[bucket].append(sw)

    wer_by_bucket = {}
    for bucket, wers_list in bucket_wers.items():
        if wers_list:
            wer_by_bucket[f"wer_{bucket}"] = round(np.mean(wers_list), 4)
            wer_by_bucket[f"count_{bucket}"] = len(wers_list)
        else:
            wer_by_bucket[f"wer_{bucket}"] = None
            wer_by_bucket[f"count_{bucket}"] = 0

    metrics = {
        "model_name": model_name,
        "num_samples": len(refs),
        # corpus-level WER (total edit ops / total reference words)
        "wer": round(overall_wer, 4),
        # mean of per-sample WER – different from corpus-level on imbalanced sets
        "wer_mean_per_sample": round(wer_mean_per_sample, 4),
        "cer": round(overall_cer, 4),
        "rtf": round(rtf, 4),
        "avg_latency_sec": round(avg_latency, 4),
        "p50_latency_sec": round(p50_latency, 4),
        "p95_latency_sec": round(p95_latency, 4),
        "total_audio_hours": round(total_audio_sec / 3600, 3),
        "total_inference_sec": round(total_inference_sec, 2),
        "gpu_memory_peak_mb": round(max_gpu_mem, 1),
        "gpu_memory_avg_mb": round(avg_gpu_mem, 1),
        "substitution_rate": round(sub_rate, 4),
        "deletion_rate": round(del_rate, 4),
        "insertion_rate": round(ins_rate, 4),
        "total_words_ref": total_words,
        "substitutions": wo.substitutions,
        "deletions": wo.deletions,
        "insertions": wo.insertions,
        "hits": wo.hits,
        **wer_by_bucket,
    }

    return metrics


def compute_per_sample_results(
    references: list[str],
    hypotheses: list[str],
    audio_paths: list[str],
    durations: list[float],
    inference_times: list[float],
    gpu_memory_peaks: list[float],
    model_name: str = "",
    sample_metadata: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """Compute per-sample metrics and return as DataFrame with full text details."""
    rows = []
    for idx, (ref, hyp, path, dur, t, g) in enumerate(zip(
        references, hypotheses, audio_paths, durations, inference_times, gpu_memory_peaks
    )):
        ref_norm = normalize_for_eval(ref)
        hyp_norm = normalize_for_eval(hyp)

        if not ref_norm:
            continue

        try:
            sample_wer = wer([ref_norm], [hyp_norm])
            sample_cer = cer([ref_norm], [hyp_norm])
        except Exception:
            sample_wer = 1.0
            sample_cer = 1.0

        rtf = t / max(dur, 0.001)
        is_correct = (ref_norm == hyp_norm)
        metadata = sample_metadata[idx] if sample_metadata and idx < len(sample_metadata) else {}
        source_audio_path = metadata.get("source_audio_path", path)

        rows.append({
            "model_name": model_name,
            "audio_path": path,
            "audio_filename": Path(path).name,
            "source_audio_path": source_audio_path,
            "source_audio_filename": Path(source_audio_path).name,
            "augmentation": metadata.get("augmentation", "clean"),
            "duration_sec": round(dur, 3),
            "duration_bucket": get_duration_bucket(dur),
            "reference_original": ref,
            "reference_normalized": ref_norm,
            "hypothesis_raw": hyp,
            "hypothesis_normalized": hyp_norm,
            "is_correct": is_correct,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
            "rtf": round(rtf, 4),
            "inference_time_sec": round(t, 4),
            "gpu_memory_peak_mb": round(g, 1),
        })

    return pd.DataFrame(rows)


def save_results(
    metrics_list: list[dict],
    per_sample_dfs: list[pd.DataFrame],
    output_dir: Optional[Path] = None,
):
    """Save aggregated and per-sample results to CSV and Excel."""
    output_dir = output_dir or config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df_agg = None
    df_samples = None

    # --- CSV: Aggregated results (append to history) ------
    if metrics_list:
        df_agg = pd.DataFrame(metrics_list)
        agg_path = output_dir / "results.csv"
        if agg_path.exists():
            try:
                df_old = pd.read_csv(agg_path)
                df_agg = pd.concat([df_old, df_agg], ignore_index=True)
                logger.info(f"Appending to existing results ({len(df_old)} previous rows)")
            except Exception as e:
                logger.warning(f"Could not read previous results, overwriting: {e}")
        df_agg.to_csv(agg_path, index=False)
        logger.info(f"Aggregated results saved: {agg_path} ({len(df_agg)} total rows)")

        # Pretty print
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        from tabulate import tabulate
        display_cols = [
            "model_name", "run_date", "wer", "cer", "rtf",
            "avg_latency_sec", "model_benchmark_wall_sec",
            "model_load_wall_sec", "inference_wall_sec", "gpu_memory_peak_mb",
            "substitution_rate", "deletion_rate", "insertion_rate",
        ]
        existing_cols = [c for c in display_cols if c in df_agg.columns]
        print(tabulate(df_agg[existing_cols], headers="keys", tablefmt="grid", showindex=False))
        print()

    # --- CSV: Per-sample results (append to history) ------
    if per_sample_dfs:
        df_samples = pd.concat(per_sample_dfs, ignore_index=True)
        samples_path = output_dir / "per_sample_results.csv"
        if samples_path.exists():
            try:
                df_old_samples = pd.read_csv(samples_path)
                df_samples = pd.concat([df_old_samples, df_samples], ignore_index=True)
            except Exception:
                pass
        df_samples.to_csv(samples_path, index=False)
        logger.info(f"Per-sample results saved: {samples_path} ({len(df_samples)} total rows)")

        # --- CSV: by-augmentation summary -----------------
        if "augmentation" in df_samples.columns:
            aug_summary = (
                df_samples.groupby(["model_name", "augmentation"])
                .agg(
                    count=("wer", "count"),
                    wer_corpus=("wer", lambda s: s.mean()),   # approx corpus-level
                    wer_mean_per_sample=("wer", "mean"),
                    cer_mean=("cer", "mean"),
                    rtf_mean=("rtf", "mean"),
                )
                .reset_index()
                .round(4)
            )
            aug_path = output_dir / "by_augmentation.csv"
            if aug_path.exists():
                try:
                    df_old_aug = pd.read_csv(aug_path)
                    aug_summary = pd.concat([df_old_aug, aug_summary], ignore_index=True)
                except Exception:
                    pass
            aug_summary.to_csv(aug_path, index=False)
            logger.info(f"By-augmentation summary saved: {aug_path}")

    # --- Excel: Full report -------------------------------
    _save_excel_report(df_agg, df_samples, output_dir)

    return output_dir


def _save_excel_report(
    df_agg: Optional[pd.DataFrame],
    df_samples: Optional[pd.DataFrame],
    output_dir: Path,
):
    """
    Save detailed Excel report with multiple sheets:
      - Summary: aggregated model metrics
      - Details: per-sample with full reference & hypothesis text
      - Errors: worst predictions (highest WER) for analysis
      - Correct: perfectly recognized samples
    """
    try:
        excel_path = output_dir / "benchmark_report.xlsx"
        logger.info(f"Writing Excel report: {excel_path}")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # --- Sheet 1: Summary -------------------------
            if df_agg is not None and not df_agg.empty:
                df_agg.to_excel(writer, sheet_name="Summary", index=False)

                # Auto-adjust column widths
                ws = writer.sheets["Summary"]
                for col_idx, col in enumerate(df_agg.columns, 1):
                    max_len = max(
                        len(str(col)),
                        df_agg[col].astype(str).str.len().max() if len(df_agg) > 0 else 0
                    )
                    ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 3, 40)

            # --- Sheet 2: Details (per-sample) ------------
            if df_samples is not None and not df_samples.empty:
                # Reorder columns for readability: text fields first
                detail_cols = [
                    "model_name",
                    "augmentation",
                    "audio_filename",
                    "source_audio_filename",
                    "duration_sec",
                    "reference_original",
                    "reference_normalized",
                    "hypothesis_raw",
                    "hypothesis_normalized",
                    "is_correct",
                    "wer",
                    "cer",
                    "rtf",
                    "inference_time_sec",
                    "gpu_memory_peak_mb",
                    "duration_bucket",
                    "audio_path",
                    "source_audio_path",
                ]
                existing_detail_cols = [c for c in detail_cols if c in df_samples.columns]
                df_details = df_samples[existing_detail_cols].copy()
                df_details.to_excel(writer, sheet_name="Details", index=False)

                ws = writer.sheets["Details"]
                # Set reasonable widths for text columns
                text_col_width = {
                    "reference_original": 50,
                    "reference_normalized": 50,
                    "hypothesis_raw": 50,
                    "hypothesis_normalized": 50,
                    "audio_filename": 25,
                    "source_audio_filename": 25,
                    "augmentation": 18,
                    "model_name": 30,
                    "audio_path": 60,
                    "source_audio_path": 60,
                }
                for col_idx, col in enumerate(existing_detail_cols, 1):
                    width = text_col_width.get(col, 15)
                    ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = width

                # --- Sheet 3: Errors (worst predictions) --
                df_errors = df_samples[df_samples["wer"] > 0].sort_values(
                    "wer", ascending=False
                ).head(100)
                if not df_errors.empty:
                    df_errors[existing_detail_cols].to_excel(
                        writer, sheet_name="Errors (worst)", index=False
                    )

                # --- Sheet 4: Correct predictions ---------
                df_correct = df_samples[df_samples["is_correct"]]
                if not df_correct.empty:
                    df_correct[existing_detail_cols].head(100).to_excel(
                        writer, sheet_name="Correct", index=False
                    )

                # --- Sheet 5: Per-model stats -------------
                if "model_name" in df_samples.columns:
                    stats_rows = []
                    for model, group in df_samples.groupby("model_name"):
                        stats_rows.append({
                            "model_name": model,
                            "total_samples": len(group),
                            "correct_count": group["is_correct"].sum(),
                            "accuracy_%": round(group["is_correct"].mean() * 100, 2),
                            "mean_wer": round(group["wer"].mean(), 4),
                            "median_wer": round(group["wer"].median(), 4),
                            "mean_cer": round(group["cer"].mean(), 4),
                            "mean_rtf": round(group["rtf"].mean(), 4),
                            "mean_latency_sec": round(group["inference_time_sec"].mean(), 4),
                        })
                    df_stats = pd.DataFrame(stats_rows)
                    df_stats.to_excel(writer, sheet_name="Per-model Stats", index=False)

        logger.info(f"Excel report saved: {excel_path}")

    except ImportError:
        logger.warning("openpyxl not installed - skipping Excel export. pip install openpyxl")
    except Exception as e:
        logger.error(f"Failed to save Excel report: {e}")


if __name__ == "__main__":
    # Standalone test with dummy data
    refs = ["привет как дела", "сегодня хорошая погода", "доброе утро"]
    hyps = ["привет как дела", "сегодня хорошая погода в москве", "доброе утра"]
    durs = [3.5, 5.0, 2.5]
    times = [0.5, 0.8, 0.3]
    gpus = [1000.0, 1200.0, 900.0]

    metrics = compute_metrics(refs, hyps, durs, times, gpus, "test_model")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
