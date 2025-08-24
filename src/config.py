import yaml, os

DEFAULTS = {
    "seed": 42,
    "sample_rate": 16000,
    "duration_sec": 4.0,
    "mono": True,
    "feature_type": "mel",
    "n_mfcc": 40,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-3,
    "val_split": 0.2,
    "early_stopping_patience": 10,
    "model_dir": "models",
    "model_name": "cnn_melspec.keras",
    "classes": ["Pneumonia","Healthy","Asthma"]
}

def load_config(path="config.yaml"):
    cfg = DEFAULTS.copy()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
            cfg.update(data)
    if isinstance(cfg.get("mono"), str):
        cfg["mono"] = cfg["mono"].lower() == "true"
    return cfg
