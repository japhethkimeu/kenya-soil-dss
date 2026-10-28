"""
Deploy (or update) the Kenya Soil Advisor as a PUBLIC Hugging Face Space.
Visitors sign in with Hugging Face, and their questions run on their own credits.

Run from Colab or locally, from the repo root, after steps 3 and 4 have produced
the rasters:
    python space/deploy_space.py --space <owner>/kenya-soil-advisor

<owner> is your HF username (free hosting for Gradio Spaces on personal accounts).

Token (never written to the Space files):
  HF_WRITE_TOKEN      token with write access to <owner>, used here to create
                      the Space and upload files
In Colab it comes from Colab secrets (key icon); locally, from an environment
variable. The app itself needs no token: visitors sign in with Hugging Face.

Re-running uploads the current files as a new commit, so it also updates the app.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = Path(__file__).resolve().parent.parent      # repo root
SPACE_DIR = REPO / "space"

# (local path, path inside the Space)
FILES = [
    (SPACE_DIR / "app.py", "app.py"),
    (SPACE_DIR / "requirements.txt", "requirements.txt"),
    (SPACE_DIR / "README.md", "README.md"),
    (REPO / "agent" / "__init__.py", "agent/__init__.py"),
    (REPO / "agent" / "soil_tools.py", "agent/soil_tools.py"),
    (REPO / "agent" / "soil_agent.py", "agent/soil_agent.py"),
    (REPO / "config" / "indicator_thresholds.csv", "config/indicator_thresholds.csv"),
]
RASTER_GLOBS = [("outputs/maps", "*_0_30cm.tif"), ("outputs/indicators", "*_classes.tif")]


def get_secret(name):
    try:                                   # Colab secrets, when running in Colab
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<owner>/<space-name>")
    args = ap.parse_args()

    write_token = get_secret("HF_WRITE_TOKEN")
    if not write_token:
        sys.exit("Set HF_WRITE_TOKEN (Colab secret or environment variable).")

    files = list(FILES)
    for folder, pattern in RASTER_GLOBS:
        found = sorted((REPO / folder).glob(pattern))
        if not found:
            sys.exit(f"No {pattern} in {folder}: run steps 3 and 4 first.")
        files += [(p, f"{folder}/{p.name}") for p in found]
    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        sys.exit("Missing files:\n  " + "\n  ".join(missing))

    api = HfApi(token=write_token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                    private=False, exist_ok=True)
    api.update_repo_settings(args.space, repo_type="space", private=False)
    print(f"Space ready (public): {args.space}")

    print("Uploading:")
    for p, dest in files:
        print(f"  {dest:45s} {p.stat().st_size / 1e6:7.1f} MB")
    api.create_commit(
        repo_id=args.space, repo_type="space",
        operations=[CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p))
                    for p, dest in files],
        commit_message="Deploy Kenya Soil Advisor")

    # The app uses each visitor's own sign-in token, so remove any old owner token
    try:
        api.delete_space_secret(args.space, "HF_TOKEN")
        print("Removed old HF_TOKEN secret (visitors now use their own sign-in).")
    except Exception:
        pass
    print(f"\nDone. Open https://huggingface.co/spaces/{args.space} "
          "(the first build takes a few minutes).")


if __name__ == "__main__":
    main()
