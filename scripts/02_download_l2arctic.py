"""02_download_l2arctic.py — Sondear y descargar L2-ARCTIC (RQ2 cross-L1).

Primero lista los archivos de los mirrors disponibles en Hugging Face.
"""
import sys
from huggingface_hub import HfApi

CANDIDATES = [
    "KoelLabs/L2Arctic",
    "chikingsley/l2-arctic-manual-v5.0-16k",
    "NathanRoll/l2-arctic-dataset-250",
    "chikingsley/l2-arctic-release-v5.0",
]


def main():
    api = HfApi()
    for d in CANDIDATES:
        try:
            files = api.list_repo_files(d, repo_type="dataset")
            print(f"\n== {d} == ({len(files)} archivos)")
            for f in files[:12]:
                print("  ", f)
        except Exception as e:
            print(f"\n== {d} == ERROR: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
