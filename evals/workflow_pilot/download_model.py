"""Explicit public-model download for the owner's machine; never used by inference."""
import argparse
import hashlib
import json
from pathlib import Path

from guard import MODEL_REPO, MODEL_REVISION, SDK_COMMIT


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["english", "multilingual", "typed-decisions"],
                        default="english")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(2, "UNAVAILABLE: huggingface_hub is not installed; nothing downloaded.\n")
    target = args.out.resolve()
    if target.exists() and any(target.iterdir()):
        parser.exit(2, "Choose an empty output directory to prevent mixing model revisions.\n")
    prefix = "" if args.checkpoint == "english" else args.checkpoint + "/"
    patterns = [prefix + name for name in (
        "model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*")]
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=patterns,
                      local_dir=str(target), token=False)
    payload = target / prefix
    required = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    missing = [name for name in required if not (payload / name).is_file()]
    if missing:
        parser.exit(2, "INCOMPLETE_DOWNLOAD: " + ", ".join(missing) + "\n")
    files = [payload / name for name in required[:3]]
    files.extend(sorted(p for p in (payload / "tokenizer").rglob("*") if p.is_file()))
    manifest = {
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "checkpoint": args.checkpoint, "sdk_commit": SDK_COMMIT,
        "files": {str(p.relative_to(payload)): sha256_file(p)
                  for p in files},
    }
    (payload / "pilot_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "DOWNLOADED_NOT_EVALUATED", "model_dir": str(payload),
                      "manifest": str(payload / "pilot_model_manifest.json")}))


if __name__ == "__main__":
    main()
