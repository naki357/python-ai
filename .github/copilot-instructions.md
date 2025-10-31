<!-- .github/copilot-instructions.md -->
# Copilot / AI agent instructions for this repository

Purpose: Help an AI coding assistant become immediately productive in this repo by describing the architecture, developer workflows, conventions, and important integration points with concise examples.

- Big picture
  - This repo is an exploratory Python AI/data-science workspace (not a packaged library). Top-level scripts drive examples and prototypes:
    - `agentic.py` — audio / local-folder analysis, uses `librosa`, `torchaudio`, `torch`, and Hugging Face Wav2Vec2 for transcription. Contains commented Google Drive download logic.
    - `non-agentic.py` — an examples collection: scikit-learn, PyTorch, TensorFlow, Hugging Face, LangChain/RAG examples, MongoDB integration, and Google Gemini usage.
  - Data and integration points:
    - Local data: `./data/audio/`, `./data/playlists/` (scripts expect these relative paths).
    - Credentials: `./googleauth/credentials.json` and `./googleauth/credentials2.json` are used by Google OAuth flows. Running Google Drive code will generate `token.json` in the repo root.
    - External services: Hugging Face model downloads, MongoDB Atlas (see `mongoCloudURI` in `non-agentic.py`), OpenAI / Gemini (via env vars like `OPENAI_API_KEY` / `GEMINI_API_KEY`).

- Developer workflows (how to run / debug)
  - Setup (macOS / zsh):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
  - Run the main example scripts (they have example calls guarded by `if __name__ == '__main__'`):
    - Analyze local audio (default): `python agentic.py` — this will call `analyze_folder_content("./data/audio/")` as written.
    - Run the RAG demo: `python non-agentic.py` — the file currently invokes `RAG_Run_Query()` by default.
  - Quick targeted runs (preferred during development to avoid heavy model downloads):
    - Run just the audio analyzer from a shell: python -c "from agentic import analyze_folder_content; print(analyze_folder_content('./data/audio'))"
    - Import and call a single function in an interactive session or notebook to avoid running full scripts.

- Project-specific conventions and notes
  - Exploratory-first style: files are intended as runnable examples. When editing, follow this pattern:
    - Keep heavy model initialization inside functions (not at import time). If you need to add a global model, document memory and download cost.
    - Use `if __name__ == '__main__'` to protect side effects. Many example functions are defined and then selectively called in that block.
    - Plotting often calls `plt.show(block=False)` — tests or CI should switch backend (e.g., `Agg`) or avoid plotting.
  - Credentials & secrets: `googleauth/` contains OAuth client files. Do NOT commit token files or API keys. Token file `token.json` is created by the Google flow — add to `.gitignore` if missing.
  - Default paths & IDs: Google Drive folder IDs and MongoDB URIs are hard-coded in examples. If modifying, update the variables near the top of the relevant file (look for `folder_id` and `mongoCloudURI`).

- Integration & heavy dependencies
  - Models that will download large weights at runtime: Hugging Face Wav2Vec2 (`facebook/wav2vec2-base-960h`), DETR (`facebook/detr-resnet-50`), SentenceTransformer (`all-MiniLM-L6-v2`). For iterative development, mock or run on a small sample to avoid repeated downloads.
  - MongoDB Atlas: `non-agentic.py` uses `pymongo` with `mongoCloudURI`. Treat the connection string as sensitive.
  - Google Drive / OAuth: `googleapiclient` flows read `./googleauth/credentials2.json` and write `token.json`. Running these flows may open a browser (local server auth). Use CI-safe mocks for unit tests.

- Useful, concrete examples for the assistant
  - To find where the audio analysis runs: open `agentic.py` and search for `analyze_folder_content` and the bottom lines that set `audio_folder_path = "./data/audio/"`.
  - To stop heavy behavior when editing, change the `if __name__ == '__main__'` section in `non-agentic.py` to avoid invoking `RAG_Run_Query()` by default.
  - To change the Google Drive folder used by the example, update `folder_id` in `agentic.py` (commented region) or `google_drive_analysis()` in `non-agentic.py`.

- What not to change without verification
  - Do not change credential files under `googleauth/` or commit `token.json` created by OAuth. Notify the repo owner before rotating keys.
  - Avoid refactoring large model downloads into imports — keep them lazy (inside functions) to keep quick imports fast.

- Feedback
  - If any of the above is incorrect, or you want the instructions to prefer a different development workflow (e.g., Docker, Conda, tests), tell me which parts to expand or change.

-- End of copilot instructions --
