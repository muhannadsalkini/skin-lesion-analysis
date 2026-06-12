#!/usr/bin/env python3
"""Quick evaluation of all trained classification models."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Import custom layers so they can be deserialized
from src.classification_models import (
    PatchExtract, PatchEmbedding, TransformerEncoderBlock,
    build_vit_classifier, build_cnn_transformer_hybrid,
)
from src.data_pipeline import load_classification_data
from src.evaluate import get_predictions, compute_clf_metrics, count_params
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def load_model_with_custom(model_name):
    """Load model with custom objects registered."""
    custom_objects = {
        "PatchExtract": PatchExtract,
        "PatchEmbedding": PatchEmbedding,
        "TransformerEncoderBlock": TransformerEncoderBlock,
    }
    paths_to_try = [
        os.path.join(config.MODELS_DIR, model_name + "_final.keras"),
        os.path.join(config.MODELS_DIR, model_name + "_phase2_best.keras"),
        os.path.join(config.MODELS_DIR, model_name + "_best.keras"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            print("[Evaluate] Loading model from " + path)
            return tf.keras.models.load_model(path, custom_objects=custom_objects)
    raise FileNotFoundError("No saved model found for " + model_name)

_, _, test_gen, class_names, _ = load_classification_data()

models = ["custom_cnn", "efficientnetb0", "mobilenetv2", "vit_classifier", "cnn_transformer_hybrid"]
all_metrics = {}

for name in models:
    try:
        model = load_model_with_custom(name)
        params = count_params(model)
        y_true, y_pred = get_predictions(model, test_gen)
        metrics = compute_clf_metrics(y_true, y_pred)
        metrics["params_M"] = params
        # Load timing
        tp = os.path.join(config.LOGS_DIR, name + "_summary.json")
        if os.path.exists(tp):
            with open(tp) as f:
                t = json.load(f)
            metrics["time_per_epoch_s"] = t.get("time_per_epoch_s")
            metrics["total_epochs"] = t.get("total_epochs")
        all_metrics[name] = metrics
        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.save(os.path.join(config.LOGS_DIR, name + "_confusion_matrix.npy"), cm)
        print(name + ": acc=" + str(round(metrics["accuracy"]*100, 1)) +
              "% F1=" + str(round(metrics["macro_f1"], 3)) +
              " P=" + str(round(metrics["macro_precision"], 3)) +
              " R=" + str(round(metrics["macro_recall"], 3)) +
              " params=" + str(params) + "M")
    except Exception as e:
        print(name + ": ERROR - " + str(e))

out = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
with open(out, "w") as f:
    json.dump(all_metrics, f, indent=2)
print("Saved to " + out)
