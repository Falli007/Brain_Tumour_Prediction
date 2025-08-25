# ------------------------------------------------------------
# app.py — Streamlit UI for the Brain MRI Tumor Classifier
# ------------------------------------------------------------

import streamlit as st
from utils.inference import predict_file
from utils.saliency import saliency_for_file  # integrated-gradients overlay

st.set_page_config(page_title="Brain MRI Tumor Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain MRI Tumor Classifier")
st.caption("Educational demo — not for clinical use yet.")

uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.subheader("Prediction settings")

    # Keep a sensible default while you tune
    default_thr = 0.50
    thr = st.slider(
        "Decision threshold (↓ more sensitive • ↑ more specific)",
        min_value=0.05,
        max_value=0.95,
        value=default_thr,
        step=0.01,
    )

    # Start with TTA OFF while debugging; you can turn it on after you’re happy
    use_tta = st.checkbox("Use test-time augmentation (horizontal flip only)", value=False)

    # Run inference
    prob, label, img, meta = predict_file(uploaded, override_threshold=thr, use_tta=use_tta)

    st.image(img, caption=f"Prediction: {label} | Tumor probability = {prob:.2f}")
    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("Details"):
        st.json({
            "chosen_threshold": thr,
            "default_threshold_in_meta": float(meta.get("threshold", thr)),
            "classes": meta.get("classes"),
            "image_size": meta.get("img_size"),
        })

    with st.expander("Debug (developer)"):
        st.json(meta.get("debug", {}))

    with st.expander("Why did the model predict this? (Saliency)"):
        try:
            overlay, _ = saliency_for_file(uploaded)
            st.image(overlay, caption="Saliency overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute saliency: {e}")

else:
    st.info("Upload a .jpg/.png brain MRI to get a prediction.")
