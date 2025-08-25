# Brain MRI Tumor Classifier

Educational demo that loads a pretrained EfficientNetB0 model and classifies a single brain MRI slice as **tumor (yes)** or **no tumor (no)**, with a Streamlit UI and a simple saliency (input‑gradient) explanation.

> **Not for clinical use.** This project is for learning and prototyping only.

---

## Project layout

```
.
├─ app.py                  # Streamlit app
├─ smoke_test.py           # One-image CLI smoke test
├─ utils/
│  ├─ inference.py         # Rebuild model, load weights, predict (with TTA & threshold)
│  └─ saliency.py          # Input-gradient saliency + overlay
├─ model/
│  ├─ best_model.keras     # Trained weights (Keras v3 / SavedModel)
│  └─ model_meta.json      # {"classes":["no","yes"],"img_size":[224,224],"threshold":0.503}
├─ assets/                 # Optional sample images
├─ notebooks/              # Your Colab/Jupyter training notebooks
├─ requirements.txt
└─ README.md
```

---

## Setup

**Python:** 3.10–3.12 (CPU TensorFlow is fine)

```bash
# 1) (recommended) create & activate a virtual env
python -m venv .venv

# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt
```

If PowerShell blocks activation, run as current user (once):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Quick CLI smoke test

Verifies that the environment and weights/metadata load correctly.

```bash
python smoke_test.py
```

Example output:
```
OK  prob=0.53  label=yes  threshold=0.503
```

---

## Run the Streamlit app

```bash
streamlit run app.py
```

In your browser:

1. Upload a JPG/PNG MRI slice (axial view works best).
2. Adjust the **Decision threshold**  
   • Lower → more sensitive (fewer false negatives)  
   • Higher → more specific (fewer false positives)  
3. Optionally enable **Test-time augmentation** (simple horizontal flip vote).
4. See the **saliency overlay** (“Why did the model predict this?”).
<img width="1238" height="1155" alt="image" src="https://github.com/user-attachments/assets/5534592c-6f8f-493f-95b5-727aca8be3e2" />
<img width="1077" height="1129" alt="image" src="https://github.com/user-attachments/assets/610e1d2c-ebb8-4537-9ea3-ae747bd702bb" />

---

## How inference works

`utils/inference.py` rebuilds the exact inference graph and loads weights:

```
RGB (224×224) 
  → keras.applications.efficientnet.preprocess_input
  → EfficientNetB0 (include_top=False, weights=None)
  → GlobalAveragePooling2D
  → Dense(128, relu)
  → Dropout(0.5)
  → Dense(1, sigmoid)
```

- Weights come from `model/best_model.keras`.
- `model/model_meta.json` supplies:
  - `"classes"`: `["no","yes"]`
  - `"img_size"`: `[224,224]`
  - `"threshold"`: default decision threshold from validation ROC (e.g., Youden’s J)

The app slider can override the threshold at runtime.

---

## Replace the model after retraining

1. Train a compatible head (same layer sizes/order as above).
2. Export weights as **Keras v3** file (e.g., `best_model.keras`).
3. Create/update `model_meta.json`, for example:
   ```json
   { "classes": ["no","yes"], "img_size": [224,224], "threshold": 0.503 }
   ```
4. Put both files in `model/` (replacing existing ones).
5. Restart the app.

> If you change the head architecture, update `build_inference_model` in `utils/inference.py` to match exactly, or weight loading will fail.

---

## Training notebooks

Place your Colab/Jupyter notebooks in `notebooks/`.

Suggested outline:
1. Load dataset; split train/val/test.
2. Build the same model head as in `inference.py`.
3. Train with callbacks: `ModelCheckpoint` (monitor val AUC), `EarlyStopping`, `ReduceLROnPlateau`.
4. Save the best model to `best_model.keras`.
5. Compute a decision threshold on the validation ROC (e.g., Youden’s J) and store it in `model_meta.json`.

---

## Troubleshooting

- **ModuleNotFoundError (numpy/tensorflow/streamlit):** Ensure the virtual env is active and `pip install -r requirements.txt` completed successfully.
- **Shape/weight mismatch:** Your trained head differs from the one in `build_inference_model`. Re-export with the same architecture or update the code to match.
- **Large oneDNN / CPU instruction notices:** Informational; safe to ignore on CPU.
- **App loads but predictions look off:** Try enabling TTA, verify `img_size` in `model_meta.json`, and confirm your training preprocessing matches `preprocess_input` at inference.

---

## Saliency explanation

The saliency overlay highlights pixels where small input changes most affect the output. It is a **rough** explanation signal, not a clinical heatmap. Use it to sanity‑check that the model looks at plausible regions (e.g., lesion rather than borders or artifacts).

---

## License

MIT (or your preferred license).

---

## Acknowledgements

- TensorFlow / Keras
- Streamlit
- EfficientNet (B0 backbone)
