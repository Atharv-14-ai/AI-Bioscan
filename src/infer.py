import argparse, json, numpy as np
from tensorflow.keras.models import load_model
from .config import load_config
from .audio_utils import load_audio, compute_melspec, compute_mfcc_stack
from .dataset import target_frames, pad_or_trim

def load_labels(path):
    with open(path,"r") as f:
        m = json.load(f)
    return [m[str(i)] if isinstance(list(m.keys())[0], str) else m[i] for i in range(len(m))]

def preprocess(path, cfg):
    y, sr = load_audio(path, sr=cfg["sample_rate"], duration_sec=cfg["duration_sec"], mono=cfg["mono"])
    if cfg["feature_type"] == "mel":
        F = compute_melspec(y, sr, cfg["n_mels"], cfg["n_fft"], cfg["hop_length"], cfg["win_length"])
    else:
        F = compute_mfcc_stack(y, sr, cfg["n_mfcc"], cfg["n_fft"], cfg["hop_length"], cfg["win_length"])
    T = target_frames(cfg["sample_rate"], cfg["duration_sec"], cfg["hop_length"], cfg["n_fft"])
    F = pad_or_trim(F, T)
    return np.expand_dims(F, (-1,0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--labels", type=str, default="models/label_map.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or f"{cfg['model_dir']}/{cfg['model_name']}"
    model = load_model(model_path, compile=False)
    labels = load_labels(args.labels)

    x = preprocess(args.audio, cfg)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    print(f"Prediction: {labels[pred]}")
    print("Probabilities:", {labels[i]: float(f"{probs[i]:.4f}") for i in range(len(labels))})

if __name__ == "__main__":
    main()
