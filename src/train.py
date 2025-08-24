import os, argparse, json, random
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from .config import load_config
from .dataset import build_numpy

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)

def plot_debug_grid(X, y, class_names, path, n=12):
    # X: (N, H, W, 1)
    idx = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idx)/cols))
    plt.figure(figsize=(4*cols, 3*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X[k, :, :, 0], aspect='auto', origin='lower')
        plt.title(class_names[y[k]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print("Using classes (order):", cfg["classes"])
    X, y, class_names = build_numpy(args.data_dir, cfg, balance=True, aug_multiplier=2)

    # sanity checks
    counts = {class_names[i]: int((y==i).sum()) for i in range(len(class_names))}
    print("Counts after build:", counts)
    os.makedirs(cfg["model_dir"], exist_ok=True)
    plot_debug_grid(X, y, class_names, os.path.join(cfg["model_dir"], "debug_grid_before_split.png"))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["val_split"], stratify=y, random_state=cfg["seed"]
    )
    plot_debug_grid(X_val, y_val, class_names, os.path.join(cfg["model_dir"], "debug_grid_val.png"))

    # Build model
    from .model import build_strong_cnn
    model = build_strong_cnn(X_train.shape[1:], len(class_names))
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg["learning_rate"], weight_decay=1e-4)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Class weights (on train only)
    classes = np.arange(len(class_names))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {i: float(class_weights[i]) for i in range(len(class_names))}
    print("Class weights:", class_weights)

    ckpt_best = os.path.join(cfg["model_dir"], "best_model.keras")
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt_best, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=class_weights,
        callbacks=cb,
        verbose=2
    )

    # Evaluate
    y_prob = model.predict(X_val, verbose=0)
    y_pred = y_prob.argmax(1)
    print("\\nConfusion Matrix (val):\\n", confusion_matrix(y_val, y_pred))
    print("\\nReport (val):\\n", classification_report(y_val, y_pred, target_names=class_names, digits=4))

    # Save
    model.save(os.path.join(cfg["model_dir"], cfg["model_name"]))
    with open(os.path.join(cfg["model_dir"], "label_map.json"), "w") as f:
        json.dump({i:c for i,c in enumerate(class_names)}, f, indent=2)
    print("Saved models to", cfg["model_dir"])

if __name__ == "__main__":
    main()
