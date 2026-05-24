from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "indian-sign-language-isl"
DATASET_HANDLE = "prathumarikeri/indian-sign-language-isl"


def main():
    parser = ArgumentParser(description="Download the Kaggle ISL dataset for SignBridge.")
    parser.add_argument("--handle", default=DATASET_HANDLE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required. Install it with: python -m pip install kagglehub"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(args.handle, output_dir=str(output_dir))
    print(path)


if __name__ == "__main__":
    main()

