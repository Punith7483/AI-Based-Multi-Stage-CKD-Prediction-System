from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import datetime
import json

app = Flask(__name__)

# ================================
# LOAD SAVED MODEL FILES
# ================================
model = joblib.load("final/acem_model.pkl")
fed_models = joblib.load("final/federated_models.pkl")
fed_weights = joblib.load("final/federated_weights.pkl")
scaler = joblib.load("final/scaler.pkl")
feature_names = joblib.load("final/features.pkl")
X_train = np.load("final/X_train.npy")

# ================================
# LOAD MODEL ACCURACY SCORES
# ================================
with open("final/scores.json", "r") as f:
    scores = json.load(f)

acem_accuracy = scores.get("ACEM Hybrid", 0) * 100
fed_accuracy = scores.get("Federated Learning", 0) * 100

class_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]


# ================================
# SAFE FLOAT CONVERSION
# ================================
def safe_float(value):
    try:
        if value is None or str(value).strip() == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


# ================================
# FEDERATED PREDICTION FUNCTION
# ================================
def federated_predict(models, weights, X):
    probs = np.zeros((X.shape[0], 5))

    for weight, model in zip(weights, models):
        probs += weight * model.predict_proba(X)

    return probs


# ================================
# COMMON GRAPH PATHS
# ================================
def common_graph_paths():
    return {
        "individual_graph": "static/individual_ml_accuracy.png",
        "overall_graph": "static/overall_model_comparison.png",
        "federated_graph": "static/federated_client_accuracy.png",
        "confusion_graph": "static/confusion_matrix_acem.png",
        "privacy_graph": "static/privacy_vs_accuracy.png",
        "confusion_metrics_comparison_graph": "static/confusion_metrics_comparison.png",
        "common_ml_comparison_graph": "static/common_ml_accuracy_comparison.png"
    }


# ================================
# HOME PAGE
# ================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        values={},
        **common_graph_paths()
    )


# ================================
# PREDICTION ROUTE
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    values = {}

    for feature in feature_names:
        value = request.form.get(feature)
        values[feature] = value if value not in [None, ""] else "0"

    data = [safe_float(values[feature]) for feature in feature_names]
    scaled_data = scaler.transform([data])

    pred = int(model.predict(scaled_data)[0])
    prob = float(np.max(model.predict_proba(scaled_data))) * 100

    fed_probs = federated_predict(fed_models, fed_weights, scaled_data)
    fed_pred = int(np.argmax(fed_probs))
    fed_prob = float(np.max(fed_probs)) * 100

    return render_template(
        "index.html",
        values=values,
        central_result=(
            f"ACEM Hybrid Prediction: {class_names[pred]} | "
            f"Confidence: {prob:.2f}% | "
            f"Model Accuracy: {acem_accuracy:.2f}%"
        ),
        fed_result=(
            f"Federated Prediction: {class_names[fed_pred]} | "
            f"Confidence: {fed_prob:.2f}% | "
            f"Model Accuracy: {fed_accuracy:.2f}%"
        ),
        **common_graph_paths()
    )


# ================================
# EXPLAINABILITY ROUTE
# ================================
@app.route("/explain", methods=["POST"])
def explain():
    values = {}

    for feature in feature_names:
        value = request.form.get(feature)
        values[feature] = value if value not in [None, ""] else "0"

    data = [safe_float(values[feature]) for feature in feature_names]
    scaled_data = scaler.transform([data])

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    explanation = explainer.explain_instance(
        scaled_data[0],
        model.predict_proba,
        num_features=8
    )

    fig = explanation.as_pyplot_figure()
    path = "static/lime.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return render_template(
        "index.html",
        values=values,
        lime_graph=path + "?t=" + str(datetime.datetime.now().timestamp()),
        **common_graph_paths()
    )


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.run(debug=True)