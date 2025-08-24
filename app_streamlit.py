import streamlit as st, numpy as np, json, librosa, soundfile as sf, tempfile
from tensorflow.keras.models import load_model
from src.config import load_config
from src.infer import preprocess

st.set_page_config(page_title="AI-BioScan (Mel-Spec)", page_icon="", layout="centered")
st.title("AI-BioScan")
st.caption("Mel-Spectrogram CNN (Pneumonia / Healthy / Asthma)")

@st.cache_resource
def load_assets():
    cfg = load_config("config.yaml")
    model = load_model(f"{cfg['model_dir']}/{cfg['model_name']}", compile=False)
    with open(f"{cfg['model_dir']}/label_map.json","r") as f:
        label_map = json.load(f)
    labels = [label_map[str(i)] if isinstance(list(label_map.keys())[0], str) else label_map[i] for i in range(len(label_map))]
    return cfg, model, labels

try:
    cfg, model, labels = load_assets()
except Exception as e:
    st.error("Model not found. Train first to create models and label_map.json")
    st.stop()

file = st.file_uploader("Upload cough/breath audio", type=["wav","mp3","flac","ogg"])
if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        data, sr = librosa.load(file, sr=cfg["sample_rate"], mono=cfg["mono"])
        sf.write(tmp.name, data, cfg["sample_rate"])
        x = preprocess(tmp.name, cfg)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    st.subheader(f"Prediction: **{labels[pred]}**")
    st.write({labels[i]: float(f"{probs[i]:.4f}") for i in range(len(labels))})
