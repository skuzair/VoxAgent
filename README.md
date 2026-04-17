# Voice-Controlled Local AI Agent

A clean Streamlit demo that accepts microphone or uploaded audio, transcribes it with Groq Whisper, classifies intent with a Groq LLM, executes safe local tools inside `output/`, and shows each pipeline stage in a structured UI.

## Stack

- STT: Groq Whisper `whisper-large-v3`
- LLM: Groq `mixtral-8x7b-32768` by default
- UI: Streamlit
- Backend: Python
- Memory: `st.session_state`
- Tools: native Python file handling inside an `output/` sandbox

## Features

- Microphone input through `streamlit-mic-recorder`
- Audio file upload for `.wav`, `.mp3`, `.m4a`, `.ogg`, and `.webm`
- Transcript review and manual correction before intent classification
- Compact structured JSON intent payloads using `type` and `steps`
- Validation and safe fallback when LLM output is malformed
- Human-in-the-loop confirmation before file operations
- File overwrite conflict detection
- Session history stored in memory for the active Streamlit session
- Optional benchmarking script that stays outside the main app flow
- Benchmark artifacts saved automatically to `benchmarks/results/`

## Project Structure

```text
voice-agent/
|-- app.py
|-- benchmark.py
|-- agent/
|   |-- __init__.py
|   |-- intent.py
|   |-- memory.py
|   |-- stt.py
|   |-- tools.py
|   `-- validation.py
|-- benchmarks/
|   |-- audio/
|   `-- results/
|-- output/
|   `-- .gitkeep
|-- prompts/
|   |-- codegen_prompt.txt
|   |-- intent_prompt.txt
|   `-- summarize_prompt.txt
|-- .env
|-- requirements.txt
`-- README.md
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your API keys to `.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_LLM_MODEL=mixtral-8x7b-32768
GROQ_FALLBACK_MODEL=llama3-70b-8192
GROQ_LIGHTWEIGHT_MODEL=llama3-8b-8192
```

4. Start the app:

```bash
streamlit run app.py
```

## Main App Flow

1. Audio is captured from the microphone or uploaded from disk.
2. Groq Whisper transcribes the audio.
3. The transcript is shown in an editable review box so STT mistakes can be corrected.
4. The reviewed transcript is classified into a compact intent payload.
5. A validation layer checks JSON shape, required fields, and safe defaults.
6. File actions pause for confirmation and warn if a file already exists.
7. The selected local tool runs inside `output/`.
8. The UI shows the transcription, intent, action, output, and session history.

## Safety and Robustness

- All file operations are restricted to the local `output/` directory.
- Absolute paths are rejected.
- Path traversal attempts like `../../secret.txt` are blocked.
- Transcript review acts as a fail-safe before intent execution.
- Invalid or incomplete intent payloads fall back safely to chat mode.
- Existing files are not overwritten unless explicitly allowed in the confirmation step.
- Friendly UI messages are shown for STT, LLM, and file execution failures.

## Benchmarking

Benchmarking is intentionally isolated from the Streamlit app.

- Run the full benchmark:

```bash
python benchmark.py
```

- This now saves results automatically to:

- `benchmarks/results/latest.json`
- `benchmarks/results/latest_llm.csv`
- `benchmarks/results/latest_stt.csv`

- Optional custom output paths:

```bash
python benchmark.py --save-json benchmarks/results/run1.json --save-csv benchmarks/results/run1_llm.csv --save-stt-csv benchmarks/results/run1_stt.csv
```

- Use a custom audio folder:

```bash
python benchmark.py --audio-dir benchmarks/audio
```

Benchmark input is now only:

- an audio file
- a matching `.txt` file containing the true transcript

Example:

- `benchmarks/audio/sample1.wav`
- `benchmarks/audio/sample1.txt`
- `benchmarks/audio/sample2.wav`
- `benchmarks/audio/sample2.txt`
- `benchmarks/audio/sample3.flac`
- `benchmarks/audio/sample3.txt`

By default the benchmark labels these sample names as:

- `sample1` -> `clean`
- `sample2` -> `complex`
- `sample3` -> `noisy`

The script uses those reference transcripts to:

1. compare Groq Whisper transcript quality against the true transcript
2. measure STT similarity and exact-match behavior
3. run the 3 LLM models on the true transcript
4. compare LLM intent accuracy, JSON validity, and structure quality

For LLM benchmarking, expected intents are derived from the reference transcript so you do not need any extra benchmark label files.

The benchmark compares these LLMs:

- `llama3-70b-8192`
- `mixtral-8x7b-32768`
- `llama3-8b-8192`

Metrics:

- STT:
  - latency
  - transcript similarity
  - exact match
- LLM:
  - latency
  - intent accuracy
  - JSON validity
  - structure quality

## Why Groq Instead of Fully Local STT

The original objective preferred a local speech-to-text model. This demo uses Groq Whisper because it provides fast, reliable transcription without requiring heavy local model inference or GPU setup. That keeps the demo responsive on machines that may not run Whisper locally at acceptable latency.
