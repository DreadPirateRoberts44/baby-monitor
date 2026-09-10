"""Copies the exported TFLite models + label files from the repo root's
output/ into pi/models/, matching what settings.py expects. Run this from
the repo root's Python environment (needs the training-side config.py)
after (re)running export_tflite.py and export_yamnet_embedding_tflite.py.

    python pi/deploy_models.py

Then copy the whole pi/ directory (including the now-populated
pi/models/) onto the Raspberry Pi.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config  # noqa: E402

PI_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FILES_TO_COPY = [
    config.YAMNET_EMBEDDING_TFLITE_PATH,
    config.TFLITE_PATH,
    config.LABELS_PATH,
    config.CRY_REASON_TFLITE_PATH,
    config.CRY_REASON_LABELS_PATH,
]


def main():
    os.makedirs(PI_MODEL_DIR, exist_ok=True)
    missing = [f for f in FILES_TO_COPY if not os.path.exists(f)]
    if missing:
        print("Missing exported files -- run these first:")
        print("  python export_tflite.py")
        print("  python export_yamnet_embedding_tflite.py")
        print("Missing:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    for src in FILES_TO_COPY:
        dst = os.path.join(PI_MODEL_DIR, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    print(f"\nDone. pi/models/ is ready to deploy (copy the pi/ directory to the Pi).")


if __name__ == "__main__":
    main()
