from flask import Flask, request, jsonify
import os, json, numpy as np, yaml
from tensorflow.keras.models import load_model
from src.infer import preprocess_for_infer
from src.utils import load_config

app = Flask(__name__)

cfg = load_config("config.yaml")
model = None
label_map = None

def load_artifacts():
    global model, label_map
    model_path = os.path.join(cfg['model_dir'], cfg['model_name'])
    label_map_path = os.path.join(cfg['model_dir'], 'label_map.json')
    if os.path.exists(model_path):
        model = load_model(model_path)
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            idx2class = json.load(f)
        label_map = {int(k): v for k, v in idx2class.items()}

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "file field missing"}), 400
    f = request.files['file']
    tmp_path = os.path.join("models", "_tmp.wav")
    f.save(tmp_path)

    X = preprocess_for_infer(tmp_path, cfg)
    if model is None:
        load_artifacts()
    if model is None or label_map is None:
        return jsonify({"error": "Model not ready. Train first and place files in models/"}), 500
    probs = model.predict(X)[0]
    pred_idx = int(np.argmax(probs))
    return jsonify({
        "prediction": label_map[pred_idx],
        "probs": {label_map[i]: float(p) for i, p in enumerate(probs)}
    })

if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
