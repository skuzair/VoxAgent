import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, List, Tuple

from agent import BENCHMARK_LLM_MODELS, ROOT_DIR
from agent.intent import generate_intent_response, parse_intent_response
from agent.stt import transcribe_audio

BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
DEFAULT_AUDIO_DIR = BENCHMARKS_DIR / "audio"
RESULTS_DIR = BENCHMARKS_DIR / "results"
DEFAULT_JSON_RESULTS = RESULTS_DIR / "latest.json"
DEFAULT_LLM_CSV_RESULTS = RESULTS_DIR / "latest_llm.csv"
DEFAULT_STT_CSV_RESULTS = RESULTS_DIR / "latest_stt.csv"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def transcript_similarity(reference: str, candidate: str) -> float:
    return round(SequenceMatcher(None, normalize_text(reference), normalize_text(candidate)).ratio(), 3)


def derive_expected_intents(transcript: str) -> List[str]:
    text = normalize_text(transcript)
    intents: List[str] = []

    if "summarize" in text or "summary" in text:
        intents.append("summarize")

    file_words = ("create a file", "make a file", "save it to", "save to", "folder", "directory")
    code_words = ("function", "class", "script", "python file", "write code", "code")

    if any(word in text for word in file_words):
        intents.append("create_file")

    if any(word in text for word in code_words):
        if "write_code" not in intents:
            intents.append("write_code")

    if not intents:
        intents.append("chat")

    if "write_code" in intents and "create_file" in intents:
        return ["create_file", "write_code"]

    if "summarize" in intents and "create_file" in intents:
        return ["summarize", "create_file"]

    return intents


def load_benchmark_items(audio_dir: Path) -> List[dict]:
    items: List[dict] = []
    for audio_file in sorted(path for path in audio_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS):
        transcript_file = audio_file.with_suffix(".txt")
        if not transcript_file.exists():
            continue

        reference_transcript = transcript_file.read_text(encoding="utf-8").strip()
        items.append(
            {
                "name": audio_file.stem,
                "audio_path": audio_file,
                "reference_transcript": reference_transcript,
                "expected_intents": derive_expected_intents(reference_transcript),
            }
        )
    return items


def get_audio_profile_label(stem: str) -> str:
    profile_map = {
        "sample1": "clean",
        "sample2": "complex",
        "sample3": "noisy",
    }
    return profile_map.get(stem.lower(), "unlabeled")


def transcribe_with_groq(audio_file: Path) -> Tuple[str, float, str]:
    started = perf_counter()
    try:
        transcript = transcribe_audio(audio_file.read_bytes(), audio_file.suffix.replace(".", ""))
        return transcript, perf_counter() - started, ""
    except Exception as exc:  # noqa: BLE001
        return "", perf_counter() - started, str(exc)


def run_stt_benchmark(items: List[dict]) -> List[dict]:
    rows: List[dict] = []

    for item in items:
        audio_path = item["audio_path"]
        reference = item["reference_transcript"]

        groq_transcript, groq_latency, groq_error = transcribe_with_groq(audio_path)

        rows.append(
            {
                "case": item["name"],
                "file": audio_path.name,
                "audio_profile": get_audio_profile_label(audio_path.stem),
                "reference_transcript": reference,
                "expected_intents": item["expected_intents"],
                "groq_whisper_transcript": groq_transcript,
                "groq_whisper_latency_seconds": round(groq_latency, 3),
                "groq_whisper_similarity": transcript_similarity(reference, groq_transcript) if groq_transcript else 0.0,
                "groq_whisper_exact_match": normalize_text(reference) == normalize_text(groq_transcript) if groq_transcript else False,
                "groq_whisper_error": groq_error,
            }
        )

    return rows


def run_single_llm_case(transcript: str, expected_intents: List[str], model: str, case_name: str) -> Dict:
    started = perf_counter()
    json_valid = False
    structure_ok = False
    predicted_intents: List[str] = []
    warnings: List[str] = []
    error = ""

    try:
        _, raw_response = generate_intent_response(transcript, [], preferred_model=model)
        payload = parse_intent_response(raw_response, transcript)
        json_valid = True
        predicted_intents = [step.get("intent", "") for step in payload.get("steps", [])]
        warnings = payload.get("warnings", [])
        structure_ok = bool(payload.get("steps")) and not payload.get("fallback", False)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    latency = perf_counter() - started
    return {
        "case": case_name,
        "expected_intents": expected_intents,
        "predicted_intents": predicted_intents,
        "latency_seconds": round(latency, 3),
        "json_valid": json_valid,
        "structure_ok": structure_ok,
        "warnings": warnings,
        "error": error,
        "intent_match": predicted_intents == expected_intents,
    }


def run_llm_benchmark(items: List[dict], models: List[str]) -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    for model in models:
        case_rows = []
        latencies = []
        intent_matches = 0
        valid_json_count = 0
        structure_ok_count = 0

        for item in items:
            row = run_single_llm_case(
                item["reference_transcript"],
                item["expected_intents"],
                model,
                item["name"],
            )
            latencies.append(row["latency_seconds"])
            if row["intent_match"]:
                intent_matches += 1
            if row["json_valid"]:
                valid_json_count += 1
            if row["structure_ok"]:
                structure_ok_count += 1
            case_rows.append(row)

        total = len(items)
        results[model] = {
            "model": model,
            "average_latency_seconds": round(mean(latencies), 3) if latencies else 0.0,
            "intent_accuracy": round(intent_matches / total, 3) if total else 0.0,
            "json_validity_rate": round(valid_json_count / total, 3) if total else 0.0,
            "structure_quality_rate": round(structure_ok_count / total, 3) if total else 0.0,
            "cases": case_rows,
        }

    return results


def print_llm_summary(results: Dict[str, dict]) -> None:
    header = f"{'Model':<24} {'Latency(s)':<12} {'Intent Acc':<12} {'JSON Valid':<12} {'Structure':<12}"
    print(header)
    print("-" * len(header))
    for model, metrics in results.items():
        print(
            f"{model:<24} "
            f"{metrics['average_latency_seconds']:<12} "
            f"{metrics['intent_accuracy']:<12} "
            f"{metrics['json_validity_rate']:<12} "
            f"{metrics['structure_quality_rate']:<12}"
        )


def print_stt_summary(rows: List[dict]) -> None:
    header = f"{'Audio':<28} {'Type':<10} {'Groq Sim':<10} {'Groq Lat':<10} {'Status':<12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        status = "ok" if not row.get("groq_whisper_error") else "error"

        print(
            f"{row['file']:<28} "
            f"{row['audio_profile']:<10} "
            f"{row['groq_whisper_similarity']:<10} "
            f"{row['groq_whisper_latency_seconds']:<10} "
            f"{status:<12}"
        )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_llm_csv(path: Path, results: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "case",
                "expected_intents",
                "predicted_intents",
                "latency_seconds",
                "json_valid",
                "structure_ok",
                "warnings",
                "error",
                "intent_match",
            ],
        )
        writer.writeheader()
        for model, metrics in results.items():
            for case in metrics["cases"]:
                writer.writerow({"model": model, **case})


def save_stt_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "file",
                "audio_profile",
                "reference_transcript",
                "expected_intents",
                "groq_whisper_transcript",
                "groq_whisper_latency_seconds",
                "groq_whisper_similarity",
                "groq_whisper_exact_match",
                "groq_whisper_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark STT and LLM models using audio files plus reference transcripts.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="Directory containing audio files and matching .txt reference transcripts.")
    parser.add_argument("--save-json", default=str(DEFAULT_JSON_RESULTS), help="Path to save JSON results.")
    parser.add_argument("--save-csv", default=str(DEFAULT_LLM_CSV_RESULTS), help="Path to save LLM CSV results.")
    parser.add_argument("--save-stt-csv", default=str(DEFAULT_STT_CSV_RESULTS), help="Path to save STT CSV results.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio benchmark directory not found: {audio_dir}")

    items = load_benchmark_items(audio_dir)
    if not items:
        raise RuntimeError("No benchmark items found. Add audio files with matching .txt reference transcripts.")

    stt_results = run_stt_benchmark(items)
    llm_results = run_llm_benchmark(items, BENCHMARK_LLM_MODELS)
    payload = {
        "benchmark_input": "audio + matching reference transcript txt files",
        "stt_results": stt_results,
        "llm_results": llm_results,
    }

    print("STT Benchmark")
    print_stt_summary(stt_results)
    print()
    print("LLM Benchmark")
    print_llm_summary(llm_results)

    save_json(Path(args.save_json), payload)
    save_llm_csv(Path(args.save_csv), llm_results)
    save_stt_csv(Path(args.save_stt_csv), stt_results)

    print()
    print("Saved benchmark artifacts:")
    print(f"- JSON: {Path(args.save_json)}")
    print(f"- LLM CSV: {Path(args.save_csv)}")
    print(f"- STT CSV: {Path(args.save_stt_csv)}")


if __name__ == "__main__":
    main()
