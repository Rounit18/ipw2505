# Phase 1 Setup Notes

This skeleton is ready for local development:

- Flask API is available in `backend/app.py`.
- React/Vite app is available in `frontend/src/App.jsx`.
- ML phase entry points are available in `ml/`.
- `backend/data/mappings.json` is the canonical mapping file.

Current expected backend behavior:

- `/classes` returns 35 mapping entries.
- `/health` returns `degraded` until `backend/models/isl_cnn.pth` exists.
- `/predict` returns `503` until the model checkpoint is added.

Dataset status:

- Downloaded Kaggle dataset is present at `data/raw/indian-sign-language-isl/Indian`.
- Preprocessing has produced `42,745` original 64x64 images.
- Augmentation has produced `89,763` training-only images.
- Training should use `data/splits/train_with_augmented.csv`.

Next phase:

1. Validate real Hindi/Kannada labels.
2. Install backend dependencies with `pip install -r backend\requirements.txt`.
3. Train with `python ml\train.py`.
4. Evaluate per-class recall and confusion matrix.
