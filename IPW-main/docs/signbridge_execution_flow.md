# SignBridge Execution Flow and Prompt Plan

Source PRD: `D:/projects/rounit project/prd.md`  
Prepared: 2026-05-21

## 1. Product Direction

SignBridge should be built as a reliable local demo first, then improved into a polished assistive prototype. The strongest path is:

1. Prove the ML model works on a held-out dataset.
2. Prove it works on a real webcam, not only dataset images.
3. Make prediction stable enough for word building.
4. Add multilingual display and history.
5. Polish the UI and prepare demo/report evidence.

The PRD is already strong, but the execution should explicitly handle a few practical gaps:

- Add a "no hand / unclear hand" gate. Without it, the system may classify background frames as valid letters.
- Resolve the J/Z contradiction. The PRD says A-Z, but also says dynamic gestures like J and Z are non-goals. Decide whether v1 supports static dataset versions of J/Z or excludes them from real signing claims.
- Treat Hindi and Kannada labels carefully. English alphabet signs do not always map cleanly to native-script letters. For demo safety, store both `display_label` and `local_name`, and validate the final strings with a speaker/teacher.
- Use one canonical mapping source. Keep `backend/data/mappings.json` as the source and generate/copy the frontend copy automatically, or let the frontend fetch `/classes`.
- Make threshold behavior consistent. If the frontend has a threshold slider, either send `threshold` in `/predict` or let the backend return raw confidence and let the frontend decide visibility.
- Add real-webcam validation. Dataset accuracy is not enough because webcam lighting, hand distance, and background will differ.
- Bundle fonts locally if the app claims no internet dependency.
- Add a short calibration screen/check before live prediction: camera permission, lighting, hand inside guide box, API health, and model loaded.

## 2. Recommended MVP Scope

Build v1 around this concrete demo promise:

"A local browser app that recognizes selected static ISL alphabet/digit gestures from a laptop webcam, stabilizes predictions, builds words letter by letter, and displays the output in English plus validated Hindi/Kannada labels."

For Review-I / academic demo, keep the scope tight:

- Static one-hand gestures only.
- Localhost only.
- CPU inference only.
- Session-only history.
- No sentence grammar.
- No mobile app.
- No cloud storage.
- No video calling.

## 3. Bettered Architecture Flow

```text
User opens React app
  -> frontend calls GET /health
  -> if ok, app asks for camera permission
  -> user sees calibration/guide overlay
  -> frame captured every selected interval
  -> optional hand/ROI gate checks whether a hand is present
  -> frame/ROI sent to POST /predict
  -> backend preprocesses exactly like training pipeline
  -> model returns top prediction, confidence, optional top_k
  -> frontend applies threshold and majority vote
  -> word-builder state machine decides whether to append
  -> translation panel displays labels
  -> confirmed words move into session history
```

## 4. Phase Plan

### Phase 0 - Scope Lock and PRD Cleanup

Objective: Remove ambiguity before coding further.

Key decisions:

- Decide whether classes are 35 including static J/Z, or 33 excluding dynamic J/Z.
- Define translation schema: `english`, `hindi_label`, `kannada_label`, `description`.
- Decide threshold ownership: backend threshold, frontend threshold, or both.
- Decide no-hand strategy: add a `NO_HAND` class, use MediaPipe Hands, or use confidence/top-k gap heuristic.
- Decide font strategy: bundled local fonts for offline behavior.

Deliverables:

- `docs/decision_log.md`
- Updated `mappings.json` schema
- Final v1 class list
- Demo acceptance criteria

Exit criteria:

- No contradiction remains between goals, non-goals, class count, and UI behavior.

Prompt:

```text
Read the SignBridge PRD and act as a product + technical architect. Identify contradictions, hidden implementation risks, and unclear requirements. Produce a decision log for v1 with final choices for class count, J/Z support, translation schema, threshold handling, no-hand detection, font loading, and offline behavior. Keep the scope realistic for a local academic demo.
```

### Phase 1 - Repository and Environment Setup

Objective: Create a clean structure that lets ML, backend, and frontend evolve independently.

Suggested structure:

```text
signbridge/
  backend/
    app.py
    requirements.txt
    data/
    models/
    src/
      model.py
      predict.py
      mappings.py
      errors.py
    tests/
  frontend/
    package.json
    src/
      components/
      hooks/
      data/
      services/
      styles/
  ml/
    preprocess.py
    augment.py
    train.py
    evaluate.py
    create_mappings.py
    configs/
  docs/
```

Deliverables:

- Backend virtual environment instructions
- Frontend Vite app
- Shared config values documented
- `.env.example` files
- Basic run scripts

Exit criteria:

- Backend and frontend dev servers start independently.
- A new developer can run setup from README.

Prompt:

```text
Create the SignBridge project skeleton from the PRD. Use a monorepo with backend, frontend, ml, and docs folders. Add minimal runnable Flask and React/Vite apps, requirements/package files, env examples, and a README with local setup commands. Keep the structure ready for PyTorch inference, mappings.json, and the three API endpoints.
```

### Phase 2 - Data, Mappings, and Preprocessing

Objective: Make the dataset and labels reproducible.

Tasks:

- Validate raw dataset folder names against final class list.
- Generate `label_map.json` and `mappings.json` from one source.
- Resize images to 64x64 with the same RGB pipeline used at inference.
- Create stratified train/validation/test CSVs.
- Add augmentation only for training data.
- Save dataset statistics per class.
- Add a small visual audit grid for random samples per class.

Deliverables:

- `ml/preprocess.py`
- `ml/augment.py`
- `ml/create_mappings.py`
- `backend/data/mappings.json`
- `frontend/src/data/mappings.json` or frontend fetch from `/classes`
- `ml/reports/dataset_summary.json`

Exit criteria:

- Every class has enough samples.
- Train/test split has no leakage.
- Mappings align with model output indices.

Prompt:

```text
Implement the SignBridge data preparation pipeline. Read the final class list and raw Kaggle ISL dataset folders, validate all classes, resize RGB images to 64x64, create stratified train/validation/test CSVs, generate label_map and mappings.json, and produce a dataset summary report. Add augmentation for training images only. Ensure the same preprocessing assumptions can be reused during backend inference.
```

### Phase 3 - ML Baseline Training and Evaluation

Objective: Train a reliable baseline before integrating UI.

Tasks:

- Implement `SignBridgeCNN`.
- Train on processed dataset.
- Save best checkpoint based on validation accuracy.
- Export `isl_cnn.pth` plus metadata.
- Produce test accuracy, per-class recall, confusion matrix, and top failing classes.
- Add targeted retraining plan if recall is below target.

Betterment:

- Store model metadata next to checkpoint: class count, input size, label map hash, training date, accuracy.
- Track top-3 accuracy. It helps diagnose similar gestures.
- Include a small webcam-captured validation folder if possible.

Deliverables:

- `ml/train.py`
- `ml/evaluate.py`
- `backend/models/isl_cnn.pth`
- `backend/models/model_metadata.json`
- `ml/reports/confusion_matrix.png`
- `ml/reports/evaluation.json`

Exit criteria:

- Overall test accuracy >= 90 percent, or a documented reason and retraining plan.
- Per-class recall target checked.
- Model loads on CPU without CUDA.

Prompt:

```text
Build and train the SignBridgeCNN model from the PRD using PyTorch. Add training, validation, early stopping, checkpointing, and evaluation scripts. Save the best CPU-compatible state_dict as isl_cnn.pth with metadata. Generate accuracy, per-class recall, top-3 accuracy, confusion matrix, and a short report listing weak classes and recommended augmentation changes.
```

### Phase 4 - Inference Pipeline Hardening

Objective: Make backend prediction match training and behave safely on unclear frames.

Tasks:

- Implement base64 decode and RGB conversion.
- Resize/preprocess exactly like training.
- Load model and mappings once on startup.
- Return top prediction, confidence, and optionally top-3 predictions.
- Add no-hand/unclear behavior.
- Add structured exceptions for bad image, missing model, bad mapping, and low confidence.

Recommended API behavior:

- Backend returns raw prediction and confidence.
- If request includes `threshold`, backend can return 422 for below threshold.
- Response includes `top_k` for debugging and future UI confidence messaging.

Deliverables:

- `backend/src/predict.py`
- Unit tests for valid image, malformed image, low confidence, missing model
- Sample request script

Exit criteria:

- Same image always returns same class.
- Missing model does not crash server startup.
- Bad requests return JSON only.

Prompt:

```text
Implement the SignBridge inference module for Flask. Load the PyTorch model and mappings once, decode base64 PNG/JPEG frames, preprocess to RGB 64x64 exactly like training, run CPU inference with torch.no_grad, and return class data plus confidence and top_k predictions. Handle missing model, malformed image, and low-confidence predictions with structured JSON errors.
```

### Phase 5 - Flask API

Objective: Build a stable local API contract for the frontend.

Endpoints:

- `GET /health`
- `GET /classes`
- `POST /predict`

Tasks:

- Add CORS for localhost frontend origins.
- Add request validation.
- Add server uptime and model status.
- Add consistent response envelope.
- Add API tests.
- Add simple latency logging for `/predict`.

Deliverables:

- `backend/app.py`
- `backend/tests/test_api.py`
- `backend/README.md`

Exit criteria:

- All endpoints tested with success and failure cases.
- `/health` reports degraded mode when model is missing.
- `/predict` median local latency is measured.

Prompt:

```text
Build the Flask API for SignBridge with /health, /classes, and /predict. Use the existing inference module and mappings loader. Add CORS for local React dev origins, structured JSON responses, degraded startup when the model is missing, validation for request bodies, and pytest coverage for success and error paths.
```

### Phase 6 - Frontend Camera, Health, and Prediction Hook

Objective: Connect the browser camera to the backend safely.

Tasks:

- Build health-check startup flow.
- Implement webcam permission state.
- Add a hand guide overlay and calibration state.
- Capture frames to off-screen canvas at 64x64.
- Poll `/predict` at configurable interval.
- Implement majority vote over recent predictions.
- Handle API unreachable, low confidence, and no-hand states.

Betterment:

- Separate "latest backend response" from "stable displayed prediction".
- Add cooldown so the same held gesture does not append repeatedly.
- Pause polling when tab is hidden or camera is disabled.

Deliverables:

- `frontend/src/services/api.js`
- `frontend/src/hooks/useHealth.js`
- `frontend/src/hooks/usePrediction.js`
- `frontend/src/components/VideoFeed.jsx`
- `frontend/src/components/GestureDisplay.jsx`

Exit criteria:

- UI starts only when API health is ok.
- Camera denial shows a useful state.
- Holding one gesture produces a stable display.

Prompt:

```text
Implement the SignBridge React camera and prediction flow. Add startup health polling, camera permission handling, mirrored webcam display, off-screen 64x64 frame capture, configurable polling to POST /predict, majority-vote smoothing over the last 3 valid predictions, and clear UI states for loading, API down, camera denied, low confidence, and no hand detected.
```

### Phase 7 - Word Builder, Translation Panel, and History

Objective: Turn isolated letters into usable communication.

Recommended word-builder state machine:

```text
IDLE -> HAND_PRESENT -> STABLE_GESTURE -> APPENDED -> WAIT_FOR_CHANGE
WAIT_FOR_CHANGE -> STABLE_GESTURE when a different gesture is stable
WAIT_FOR_CHANGE -> IDLE when no hand/unclear state is detected
```

Tasks:

- Show English always.
- Toggle Hindi and Kannada.
- Implement current word.
- Implement backspace and confirm.
- Add session history with timestamps.
- Add copy-to-clipboard.
- Add keyboard shortcuts only if they are discoverable through standard button labels/tooltips.

Deliverables:

- `TranslationPanel.jsx`
- `WordBuilder.jsx`
- `HistoryLog.jsx`
- `SettingsBar.jsx`
- App-level state/context

Exit criteria:

- Repeated same held letter appends once only.
- Different stable gestures append correctly.
- Confirmed words appear in history.
- History copy works.

Prompt:

```text
Build the SignBridge communication UI around the stable prediction stream. Add TranslationPanel, WordBuilder, HistoryLog, and SettingsBar. Implement duplicate-letter debounce with a simple state machine so a held gesture appends once, then waits for hand removal or a different stable gesture. Support backspace, confirm word, timestamped session history, copy transcript, confidence threshold slider, polling interval selector, and Hindi/Kannada visibility toggles.
```

### Phase 8 - Integration, QA, and Performance Tuning

Objective: Prove the full system works under demo conditions.

Tasks:

- Run backend and frontend together.
- Test with real webcam across 10-15 gestures.
- Measure latency in browser and backend logs.
- Test low light, busy background, hand too far, and hand absent.
- Check Chrome, Firefox, and Safari if available.
- Verify scripts render correctly.
- Verify no internet dependency if that is claimed.
- Tune polling interval, smoothing window, and threshold.

Deliverables:

- `docs/qa_checklist.md`
- `docs/demo_runbook.md`
- `docs/known_limitations.md`
- Screenshots/video evidence
- Final metrics table

Exit criteria:

- End-to-end demo succeeds three times in a row.
- Known limitations are documented honestly.
- Latency and stability targets are measured, not guessed.

Prompt:

```text
Run an end-to-end QA pass for SignBridge. Start backend and frontend, exercise the complete flow from camera permission to prediction, word building, translation display, and history copy. Measure prediction latency, note browser/network errors, test unclear/no-hand states, verify local font behavior, and produce a QA checklist plus known limitations and demo runbook.
```

### Phase 9 - Final Demo, Report, and Presentation Package

Objective: Prepare submission-ready evidence and a smooth live demo.

Tasks:

- Freeze final code version.
- Add setup/run instructions.
- Prepare architecture diagram.
- Export evaluation metrics.
- Capture screenshots of every major state.
- Prepare a short demo script.
- Document limitations and future work.

Deliverables:

- Final README
- Final report sections
- Slides content
- Demo runbook
- Backup video/gif

Exit criteria:

- A fresh machine/setup can run the project from instructions.
- Demo can proceed even if live camera lighting is imperfect.
- Faculty questions about dataset, model, API, latency, and limitations have prepared answers.

Prompt:

```text
Prepare the final SignBridge academic demo package. Create a concise README, architecture explanation, setup steps, demo script, metrics summary, screenshots checklist, limitations, and future enhancements. Make the documentation honest, technically specific, and aligned with the PRD goals.
```

## 5. Critical Build Order

Use this order to avoid wasted work:

1. Resolve scope contradictions.
2. Validate mappings and class list.
3. Train/evaluate model.
4. Harden inference.
5. Build API.
6. Build camera + prediction hook.
7. Build word builder + multilingual UI.
8. Run real webcam QA.
9. Polish documentation and demo.

Do not spend serious time polishing UI until the live webcam prediction loop works with stable output.

## 6. Suggested Acceptance Gates

| Gate | Pass condition |
|---|---|
| Scope gate | Final class list and translation schema agreed |
| Data gate | All labels align with class indices and mappings |
| Model gate | Accuracy and per-class recall measured |
| API gate | Health/classes/predict tested with JSON errors |
| Frontend gate | Camera capture and prediction smoothing work |
| Word gate | Duplicate debounce and confirm/history work |
| Demo gate | Full flow runs three times consecutively |

## 7. Future Enhancements After v1

- Dynamic gestures using sequence models or landmark trajectories.
- Mobile responsive camera experience.
- Sentence-level communication and phrase suggestions.
- Text-to-speech for confirmed words.
- Better hand segmentation or MediaPipe landmark-based model.
- User-specific calibration/training mode.
- More Indian languages.
- PWA offline packaging.
- Optional local model inference in browser using ONNX Runtime Web.

