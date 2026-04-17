# Voice-Controlled Local AI Agent

This project is a voice-controlled AI agent built to satisfy the assignment requirement of accepting audio input, converting it to text, classifying user intent, executing local tools safely, and displaying the full pipeline in a clean UI.

## Objective

The goal of this system is to:

- accept audio from microphone input or uploaded files
- convert speech to text
- classify the user’s intent
- execute the appropriate local tool
- show the transcript, detected intent, action taken, and final result in the UI

## Implemented Stack

- STT: Groq Whisper `whisper-large-v3`
- LLM: Groq `mixtral-8x7b-32768` by default
- UI: Streamlit
- Backend: Python
- Memory: `st.session_state`

## Why Groq Was Used for STT

The assignment preferred a local STT model when possible. This implementation uses Groq Whisper instead of a fully local Whisper or wav2vec setup because it provides fast, reliable transcription without requiring heavy local inference hardware. This was a practical hardware workaround to keep the demo responsive and easy to run on a normal development machine. I wasnt able to use local models since I work on a laptop which doesnt have a GPU or good specs, so i had to rely on online models for good enough performance.

## System Requirements Coverage

### 1. Audio Input

The system supports both required audio input methods:

- direct microphone input using `streamlit-mic-recorder`
- uploaded audio files such as `.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`, and `.flac`

### 2. Speech-to-Text

Audio is transcribed with Groq Whisper.

The app also includes a transcript review step so the user can manually fix STT mistakes before intent classification. This helps with edge cases such as misheard words like `palindrome` being transcribed incorrectly.

### 3. Intent Understanding

The transcribed text is sent to a Groq LLM for intent classification.

Supported intents:

- create file
- write code
- summarize text
- general chat

The app uses a compact structured JSON format with validation and safe fallback behavior if the LLM returns malformed output.

### 4. Tool Execution

Based on the detected intent, the system can:

- create files or folders
- generate code and save it into a file
- summarize provided text
- answer in general chat mode

All file and code-writing operations are restricted to the local [`output/`](D:\voxagent\output) folder so the system cannot overwrite arbitrary files on the machine.

### 5. User Interface

The frontend is built with Streamlit.

The UI displays:

- the transcribed text
- the detected intent
- the action taken
- the final output
- session history

It also includes:

- loading states
- clear success, warning, and error messages
- a human-in-the-loop confirmation step before file operations

## Example Flow

User says:

`Create a Python file with a retry function.`

System flow:

1. Transcribe audio to text.
2. Show the transcript for manual review.
3. Detect intent such as `create_file` and `write_code`.
4. Generate Python code.
5. Create the file inside [`output/`](D:\voxagent\output).
6. Show the transcript, intent, action, and result in the UI.

## Bonus Features Implemented

This project includes several optional improvements from the assignment:

- compound command support
- human-in-the-loop confirmation
- graceful degradation for STT, intent parsing, and tool failures
- session memory with `st.session_state`
- model benchmarking in a separate script

## Architecture Overview

Main app flow:

1. Audio input from microphone or uploaded file
2. STT with Groq Whisper
3. Editable transcript review
4. Intent classification with Groq LLM
5. Validation of structured intent output
6. Confirmation before file operations
7. Safe tool execution in `output/`
8. UI rendering of transcript, intent, action, output, and history

Core files:

- [app.py](D:\voxagent\app.py): Streamlit UI and orchestration
- [agent/stt.py](D:\voxagent\agent\stt.py): speech-to-text logic
- [agent/intent.py](D:\voxagent\agent\intent.py): LLM intent classification
- [agent/validation.py](D:\voxagent\agent\validation.py): intent payload validation
- [agent/tools.py](D:\voxagent\agent\tools.py): local tool execution
- [agent/memory.py](D:\voxagent\agent\memory.py): session memory helpers
- [benchmark.py](D:\voxagent\benchmark.py): isolated benchmark runner

## Safety and Robustness

This project is a demo system, but it includes basic safety and resilience:

- all file writes are sandboxed to [`output/`](D:\voxagent\output)
- path traversal and absolute paths are blocked
- existing files are not overwritten unless explicitly allowed
- transcript review acts as a fail-safe before intent execution
- malformed intent output falls back safely instead of crashing
- friendly UI errors are shown for STT, LLM, and tool failures

## Benchmarking

Benchmarking is intentionally isolated from the main app and does not affect the UI pipeline.

Current benchmark coverage:

- STT benchmark with Groq Whisper against reference transcripts
- LLM benchmark across:
  - `llama3-70b-8192`
  - `mixtral-8x7b-32768`
  - `llama3-8b-8192`

Measured metrics:

- STT:
  - latency
  - transcript similarity
  - exact match
- LLM:
  - latency
  - intent accuracy
  - JSON validity
  - structure quality

Benchmark input format:

- place audio files in [`benchmarks/audio/`](D:\voxagent\benchmarks\audio)
- add a matching `.txt` file for each audio file containing the true transcript

Example:

- `benchmarks/audio/sample1.wav`
- `benchmarks/audio/sample1.txt`

Run:

```bash
python benchmark.py
```

Benchmark outputs are saved automatically in [`benchmarks/results/`](D:\voxagent\benchmarks\results).

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

## Setup Instructions

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key to [`.env`](D:\voxagent\.env):

```env
GROQ_API_KEY=your_key_here
GROQ_LLM_MODEL=mixtral-8x7b-32768
GROQ_FALLBACK_MODEL=llama3-70b-8192
GROQ_LIGHTWEIGHT_MODEL=llama3-8b-8192
```

4. Run the app:

```bash
streamlit run app.py
```
