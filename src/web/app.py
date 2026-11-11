"""Servidor web: sirve la interfaz (HTML/CSS/JS) y expone /modelos y /predict, que usan los modelos entrenados.

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

from src.preprocesamiento import CLASSES, IMG_SIZE, preprocess  # noqa: E402

MODELS_DIR = ROOT / "modelos"
MAX_UPLOAD_MB = 10
MAX_PIXELS = 50_000_000

# Modelos disponibles. "preprocesado" debe coincidir con el redimensionado con que se entreno cada uno
# ("pad": conserva la proporcion y rellena; "stretch": estira a cuadrado) y "tamano" con el lado en pixeles
# (128 si no se indica). "externas" son los aciertos
# del modelo en las 8 imagenes externas del notebook (seccion 7 en adelante).
MODELS = {
    "efficientnetb0_stretch_aug_head": {
        "nombre": "EfficientNetB0 + estirar, aumento y solo la cabeza",
        "archivo": "efficientnetb0_stretch_aug_head.keras", "preprocesado": "stretch",
        "detalle": "Modelo seleccionado: transferencia de aprendizaje, imágenes estiradas y aumento de datos; la base EfficientNetB0 permanece congelada. Aciertos en imágenes externas: 6 de 8."},
    "efficientnetb0_pad_aug": {
        "nombre": "EfficientNetB0 + aumento de datos (padding)",
        "archivo": "efficientnetb0_pad_aug.keras", "preprocesado": "pad",
        "detalle": "Transfer learning, padding y aumento de datos. Aciertos en imágenes externas: 5 de 8."},
    "efficientnetb0_stretch_aug": {
        "nombre": "EfficientNetB0 + estirar y aumento de datos",
        "archivo": "efficientnetb0_stretch_aug.keras", "preprocesado": "stretch",
        "detalle": "Transfer learning, imágenes estiradas y aumento de datos. Aciertos en imágenes externas: 5 de 8."},
    "efficientnetb0_stretch_aug_224": {
        "nombre": "EfficientNetB0 + estirar y aumento (224 px)",
        "archivo": "efficientnetb0_stretch_aug_224.keras", "preprocesado": "stretch", "tamano": 224,
        "detalle": "Transfer learning a 224 × 224, imágenes estiradas y aumento de datos. Aciertos en imágenes externas: 5 de 8."},
    "efficientnetb0_stretch_aug_dirigido": {
        "nombre": "EfficientNetB0 + estirar y aumento dirigido",
        "archivo": "efficientnetb0_stretch_aug_dirigido.keras", "preprocesado": "stretch",
        "detalle": "Transfer learning, imágenes estiradas y aumento dirigido (alejar, recortar y borrar zonas). Aciertos en imágenes externas: 4 de 8."},
    "efficientnetb0_stretch": {
        "nombre": "EfficientNetB0 + estirar",
        "archivo": "efficientnetb0_stretch.keras", "preprocesado": "stretch",
        "detalle": "Transfer learning con imágenes estiradas. Aciertos en imágenes externas: 4 de 8."},
    "efficientnetb0_pad": {
        "nombre": "EfficientNetB0 (transfer learning)",
        "archivo": "efficientnetb0_pad.keras", "preprocesado": "pad",
        "detalle": "Transfer learning con padding. Aciertos en imágenes externas: 4 de 8."},
    "efficientnetb0_dropout_pad": {
        "nombre": "EfficientNetB0 + Dropout",
        "archivo": "efficientnetb0_dropout_pad.keras", "preprocesado": "pad",
        "detalle": "Transfer learning con padding y Dropout. Aciertos en imágenes externas: 4 de 8."},
    "cnn_dropout_pad": {
        "nombre": "CNN propia + Dropout",
        "archivo": "cnn_dropout_pad.keras", "preprocesado": "pad",
        "detalle": "Red entrenada desde cero con Dropout. Aciertos en imágenes externas: 2 de 8."},
    "cnn_base_pad": {
        "nombre": "CNN propia (modelo base)",
        "archivo": "cnn_base_pad.keras", "preprocesado": "pad",
        "detalle": "Red entrenada desde cero, sin regularización. Aciertos en imágenes externas: 2 de 8."},
}
AVAILABLE = {k: v for k, v in MODELS.items() if (MODELS_DIR / v["archivo"]).exists()}
DEFAULT_MODEL = "efficientnetb0_stretch_aug_head" if "efficientnetb0_stretch_aug_head" in AVAILABLE else next(iter(AVAILABLE), None)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_loaded = {}
_lock = threading.Lock()


def predict_probabilities(model_id, x):
    """Carga el modelo la primera vez que se pide y devuelve las probabilidades de cada clase."""
    with _lock:
        if model_id not in _loaded:
            _loaded[model_id] = keras.models.load_model(MODELS_DIR / AVAILABLE[model_id]["archivo"], compile=False)
        return _loaded[model_id].predict(x[None], verbose=0)[0]


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/modelos")
def list_models():
    return jsonify(
        predeterminado=DEFAULT_MODEL,
        modelos=[{"id": k, "nombre": v["nombre"], "detalle": v["detalle"]} for k, v in AVAILABLE.items()],
    )


@app.post("/predict")
def predict():
    model_id = request.form.get("modelo", DEFAULT_MODEL)
    if model_id not in AVAILABLE:
        return jsonify(error="Modelo no válido."), 400
    file = request.files.get("imagen")
    if file is None or file.filename == "":
        return jsonify(error="No se recibio ninguna imagen."), 400
    try:
        image = Image.open(file.stream)
        if image.width * image.height > MAX_PIXELS:
            return jsonify(error="La imagen es demasiado grande."), 400
        x = preprocess(image, size=AVAILABLE[model_id].get("tamano", IMG_SIZE), mode=AVAILABLE[model_id]["preprocesado"])
    except Exception:
        return jsonify(error="No se pudo leer el archivo como imagen."), 400

    probs = predict_probabilities(model_id, x)
    best = int(probs.argmax())

    buffer = io.BytesIO()
    Image.fromarray(x).save(buffer, format="PNG")
    processed = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return jsonify(
        clase=CLASSES[best],
        probabilidades={c: float(p) for c, p in zip(CLASSES, probs)},
        imagen_procesada=processed,
        modelo=model_id,
        modelo_nombre=AVAILABLE[model_id]["nombre"],
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"La imagen supera {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    if not AVAILABLE:
        sys.exit(f"No se encontro ningun modelo en {MODELS_DIR}. Entrenalos desde el notebook primero.")
    missing = [v["archivo"] for k, v in MODELS.items() if k not in AVAILABLE]
    if missing:
        print("Aviso: faltan estos modelos y no aparecerán en la lista:", ", ".join(missing))
    app.run(host="127.0.0.1", port=8000)
