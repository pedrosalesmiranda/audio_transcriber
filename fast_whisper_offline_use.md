Yes, once the model is downloaded, everything runs fully offline.

  First run (online required):
  - faster-whisper downloads the model from Hugging Face and caches it locally
  - For "base" model: ~145 MB

  All subsequent runs (offline):
  - Model is loaded from local cache (default: ~/.cache/huggingface/hub/)
  - No internet connection needed

  To force offline mode (prevents accidental downloads):
  import os
  os.environ["TRANSFORMERS_OFFLINE"] = "1"
  os.environ["HF_DATASETS_OFFLINE"] = "1"

  Or pre-download the model explicitly:
  pip install huggingface_hub
  huggingface-cli download Systran/faster-whisper-base

  The Silero VAD model (used by vad_filter=True) is also cached locally after first use — so that works offline too.