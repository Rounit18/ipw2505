# SignBridge Data Pipeline

The Phase 2 data pipeline prepares raw Kaggle ISL class folders for model training.

Downloaded dataset path:

```text
D:\projects\rounit project\signbridge\data\raw\indian-sign-language-isl\Indian
```

Project-relative path:

```text
data/raw/indian-sign-language-isl/Indian
```

Download command:

```powershell
python -m pip install -r ml\requirements.txt
python ml\download_dataset.py
```

The raw and generated dataset folders are not committed to Git.

Current prepared dataset status:

| Item | Count |
|---|---:|
| Raw class images | 42,745 |
| Processed original images | 42,745 |
| Train rows | 29,921 |
| Validation rows | 6,412 |
| Test rows | 6,412 |
| Training-only augmented rows | 89,763 |
| Combined training rows | 119,684 |

Generated reports:

```text
ml/reports/dataset_summary.json
ml/reports/augmentation_summary.json
```

## Expected Raw Dataset Layout

```text
raw_dataset/
  A/
  B/
  ...
  Z/
  1/
  ...
  9/
```

All 35 folders are required. Extra folders fail validation unless `--allow-extra-folders` is passed.

## Preprocess

```powershell
cd "D:\projects\rounit project\signbridge"
python ml\preprocess.py
```

Default outputs:

```text
data/isl_processed/            RGB 64x64 PNG images
data/splits/all.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
ml/configs/label_map.json
ml/reports/dataset_summary.json
backend/data/mappings.json
frontend/src/data/mappings.json
```

## Preprocess With Training-Only Augmentation

```powershell
python ml\preprocess.py --augment-copies 3
```

This creates:

```text
data/isl_augmented/
data/splits/train_augmented_only.csv
data/splits/train_with_augmented.csv
ml/reports/augmentation_summary.json
```

Augmentation is applied only to records from `train.csv`. Validation and test images are never augmented.

## Preprocessing Contract

Both ML preprocessing and backend inference read:

```text
preprocessing_config.json
```

Current contract:

- RGB conversion
- Resize to 64x64
- LANCZOS resampling
- Pixel values divided by 255
- CHW tensor order for inference/training
- float32 dtype
