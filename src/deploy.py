import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID", "your-username/tomato-growth-stage")
MODEL_DIR = "outputs/models"


def deploy():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set. Cannot deploy.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"[SUCCESS] Model deployed to https://huggingface.co/{REPO_ID}")
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        sys.exit(1)


if __name__ == "__main__":
    deploy()
