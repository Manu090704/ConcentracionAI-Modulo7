"""Servidor web: sirve la interfaz (HTML/CSS/JS) y expone /predict, que usa el modelo entrenado.

Ejecutar desde la raiz del repositorio:  python src/web/app.py
y abrir http://127.0.0.1:8000
"""
import base64
import io
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402
from PIL import Image  # noqa: E402

from src.preprocesamiento import CLASSES, preprocess  # noqa: E402

MODEL_PATH = ROOT / "modelos" / "cnn_base_pad.keras"
MAX_UPLOAD_MB = 10
MAX_PIXELS = 50_000_000

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_model = None
_model_lock = threading.Lock()


def predict_probabilities(x):
    """Carga el modelo la primera vez y devuelve las probabilidades de cada clase."""
    global _model
    with _model_lock:
        if _model is None:
            _model = keras.models.load_model(MODEL_PATH)
        return _model.predict(x[None], verbose=0)[0]


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/predict")
def predict():
    file = request.files.get("imagen")
    if file is None or file.filename == "":
        return jsonify(error="No se recibio ninguna imagen."), 400
    try:
        image = Image.open(file.stream)
        if image.width * image.height > MAX_PIXELS:
            return jsonify(error="La imagen es demasiado grande."), 400
        x = preprocess(image)
    except Exception:
        return jsonify(error="No se pudo leer el archivo como imagen."), 400

    probs = predict_probabilities(x)
    best = int(probs.argmax())

    buffer = io.BytesIO()
    Image.fromarray(x).save(buffer, format="PNG")
    processed = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return jsonify(
        clase=CLASSES[best],
        probabilidades={c: float(p) for c, p in zip(CLASSES, probs)},
        imagen_procesada=processed,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"La imagen supera {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        sys.exit(f"No se encontro el modelo en {MODEL_PATH}. Entrenalo desde el notebook primero.")
    app.run(host="127.0.0.1", port=8000)
