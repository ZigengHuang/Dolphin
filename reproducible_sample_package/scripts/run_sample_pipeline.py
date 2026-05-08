#!/usr/bin/env python3
"""Run the public Dolphin sample workflow on privacy-preserving examples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import urllib.request
import wave
from pathlib import Path


EMOTION_KEYWORDS = {
    "fear": {"worried", "afraid", "anxious", "concerned", "fear", "scared"},
    "sadness": {"sad", "tired", "hopeless", "upset", "still", "pain"},
    "surprise": {"forgot", "sudden", "unexpected", "cancel", "missed"},
    "anger": {"angry", "frustrated", "unfair", "annoyed"},
    "happiness": {"relieved", "happy", "glad", "better"},
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_package_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_knowledge(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_knowledge(case: dict, knowledge_rows: list[dict], top_k: int) -> list[dict]:
    query_tokens = tokenize(
        " ".join([case.get("department", ""), case.get("speaker_query", ""), case.get("transcript", "")])
    )
    scored = []
    for row in knowledge_rows:
        row_tokens = tokenize(" ".join(row.values()))
        overlap = len(query_tokens & row_tokens)
        department_bonus = 5 if row.get("department") == case.get("department") else 0
        scored.append((overlap + department_bonus, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for score, row in scored[:top_k] if score > 0]


def read_wav_features(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV for sample audio: {path}")

    samples = []
    for i in range(0, len(raw), sample_width * channels):
        channel_values = []
        for channel in range(channels):
            start = i + channel * sample_width
            value = int.from_bytes(raw[start : start + sample_width], byteorder="little", signed=True)
            channel_values.append(value / 32768.0)
        samples.append(sum(channel_values) / len(channel_values))

    if not samples:
        return {"duration_sec": 0.0, "rms": 0.0, "peak": 0.0, "zero_crossing_rate": 0.0}

    duration = frame_count / float(sample_rate)
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    peak = max(abs(value) for value in samples)
    crossings = sum(1 for left, right in zip(samples, samples[1:]) if (left >= 0) != (right >= 0))
    zcr = crossings / max(1, len(samples) - 1)

    return {
        "duration_sec": round(duration, 3),
        "sample_rate": sample_rate,
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "zero_crossing_rate": round(zcr, 4),
    }


def summarize_audio(features: dict) -> str:
    return (
        f"duration={features['duration_sec']}s; "
        f"sample_rate={features['sample_rate']}Hz; "
        f"rms={features['rms']}; "
        f"peak={features['peak']}; "
        f"zero_crossing_rate={features['zero_crossing_rate']}"
    )


def infer_emotion(case: dict, audio_features: dict) -> dict:
    text = f"{case.get('speaker_query', '')} {case.get('transcript', '')}".lower()
    scores = {emotion: 0.0 for emotion in ["neutral", *EMOTION_KEYWORDS.keys()]}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] += sum(1.0 for keyword in keywords if keyword in text)

    if audio_features.get("rms", 0.0) > 0.22:
        scores["fear"] += 0.4
        scores["surprise"] += 0.3
    if audio_features.get("zero_crossing_rate", 0.0) < 0.04:
        scores["sadness"] += 0.3

    scores["neutral"] += 0.2
    total = sum(math.exp(value) for value in scores.values())
    probabilities = {emotion: round(math.exp(value) / total, 4) for emotion, value in scores.items()}
    predicted = max(probabilities, key=probabilities.get)
    return {"predicted_emotion": predicted, "emotion_probabilities": probabilities}


def format_knowledge(rows: list[dict]) -> str:
    if not rows:
        return "No matching sample knowledge entry was retrieved."
    return "\n".join(
        f"- {row.get('department', 'Unknown')} / {row.get('topic', 'General')}: {row.get('content', '')}"
        for row in rows
    )


def build_reasoning_path(case: dict, audio_summary: str, retrieved_knowledge: str, emotion_result: dict) -> str:
    emotion = emotion_result["predicted_emotion"]
    return (
        f"Emotion Interpretation: The sample pipeline predicts {emotion} as the dominant affective state, "
        f"using text cues together with simple audio descriptors ({audio_summary}).\n"
        f"Semantic Interpretation: The patient education need is: {case.get('communication_goal', '')}\n"
        "Response Guideline: Acknowledge the concern first, provide concise education grounded in the "
        "retrieved knowledge, avoid individualized medical decisions, and direct safety-sensitive questions "
        "to the local clinical team.\n"
        f"Retrieved Knowledge Used:\n{retrieved_knowledge}"
    )


def template_response(case: dict, reasoning_path: str, retrieved_rows: list[dict]) -> str:
    topic_sentence = retrieved_rows[0]["content"] if retrieved_rows else "Please follow the local care team's guidance."
    return (
        "I understand why this feels concerning. "
        f"{topic_sentence} "
        "Because the safest next step can depend on your exact situation, please contact the clinical team if "
        "anything is unclear, symptoms worsen, or you need individualized instructions."
    )


def call_openai_compatible(prompt: str, config: dict) -> str:
    llm_config = config.get("llm", {})
    endpoint = os.environ.get(llm_config.get("endpoint_env", "DOLPHIN_SAMPLE_LLM_ENDPOINT"), "")
    model = os.environ.get(llm_config.get("model_env", "DOLPHIN_SAMPLE_LLM_MODEL"), "")
    api_key = os.environ.get(llm_config.get("api_key_env", "DOLPHIN_SAMPLE_LLM_API_KEY"), "")
    if not endpoint or not model or not api_key:
        raise RuntimeError("OpenAI-compatible mode requires endpoint, model, and API key environment variables.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def run_pipeline(config_path: Path, output_path: Path, llm_provider: str) -> None:
    config = load_json(config_path)
    cases_path = resolve_package_path(config_path, config["sample_cases_path"])
    knowledge_path = resolve_package_path(config_path, config["knowledge_base_path"])
    reasoning_prompt_path = resolve_package_path(config_path, config["reasoning_prompt_path"])
    response_prompt_path = resolve_package_path(config_path, config["response_prompt_path"])

    cases = load_cases(cases_path)
    knowledge_rows = load_knowledge(knowledge_path)
    reasoning_prompt_template = load_text(reasoning_prompt_path)
    response_prompt_template = load_text(response_prompt_path)
    top_k = int(config.get("top_k_knowledge", 2))

    outputs = []
    for case in cases:
        audio_path = (cases_path.parent / case["audio_path"]).resolve()
        audio_features = read_wav_features(audio_path)
        audio_summary = summarize_audio(audio_features)
        retrieved_rows = retrieve_knowledge(case, knowledge_rows, top_k=top_k)
        retrieved_knowledge = format_knowledge(retrieved_rows)
        emotion_result = infer_emotion(case, audio_features)

        reasoning_prompt = reasoning_prompt_template.format(
            speaker_query=case["speaker_query"],
            transcript=case["transcript"],
            audio_summary=audio_summary,
            retrieved_knowledge=retrieved_knowledge,
        )
        reasoning_path = build_reasoning_path(case, audio_summary, retrieved_knowledge, emotion_result)

        response_prompt = response_prompt_template.format(
            speaker_query=case["speaker_query"],
            reasoning_path=reasoning_path,
            retrieved_knowledge=retrieved_knowledge,
        )
        if llm_provider == "openai-compatible":
            response = call_openai_compatible(response_prompt, config)
        else:
            response = template_response(case, reasoning_path, retrieved_rows)

        outputs.append(
            {
                "case_id": case["case_id"],
                "department": case["department"],
                "privacy_note": config.get("privacy_note", ""),
                "audio_features": audio_features,
                "emotion_result": emotion_result,
                "retrieved_knowledge": retrieved_rows,
                "reasoning_prompt": reasoning_prompt,
                "reasoning_path": reasoning_path,
                "response_prompt": response_prompt,
                "generated_response": response,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    emotion_counts = {}
    for row in outputs:
        emotion = row["emotion_result"]["predicted_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    print(f"Wrote {len(outputs)} sample outputs to {output_path}")
    print(f"Predicted emotion distribution: {emotion_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dolphin's public reproducible sample workflow.")
    parser.add_argument("--config", required=True, type=Path, help="Path to sample_config.json")
    parser.add_argument("--output", required=True, type=Path, help="Path to output JSONL")
    parser.add_argument(
        "--llm-provider",
        choices=["template", "openai-compatible"],
        default="template",
        help="Response-generation backend. Template mode is deterministic and offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config.resolve(), args.output.resolve(), args.llm_provider)


if __name__ == "__main__":
    main()
