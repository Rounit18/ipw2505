# SignBridge

Local academic prototype for static Indian Sign Language gesture recognition.

## Structure

```text
backend/   Flask API and PyTorch inference
frontend/  React + Vite browser UI
ml/        Data preparation, training, and evaluation scripts
docs/      Project notes and runbooks
```

## Local Setup

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

The backend starts in degraded mode until `backend/models/isl_cnn.pth` is available.

Data preparation:

```powershell
python -m pip install -r ml\requirements.txt
python ml\download_dataset.py
python ml\preprocess.py --augment-copies 3
```

The downloaded Kaggle class folders are expected at
`data/raw/indian-sign-language-isl/Indian`. Override with `--raw-dir` only if
you move the dataset.

Dataset files, processed images, augmented images, model checkpoints, and build
artifacts are intentionally not committed to Git. Recreate them with the commands
above.

Local prepared dataset from the development machine:

- Raw class images: `42,745`
- Processed original images: `42,745`
- Train / validation / test split: `29,921 / 6,412 / 6,412`
- Training-only augmented images: `89,763`
- Combined training CSV rows: `119,684`

See `docs/data_pipeline.md` for the full pipeline contract.

Training:

```powershell
python ml\train.py
```

Training saves the best CPU-compatible state dict to
`backend/models/isl_cnn.pth` and metadata to `backend/models/model_metadata.json`.
The default training device is `auto`, so it uses CUDA when PyTorch can see a GPU
and falls back to CPU otherwise.
