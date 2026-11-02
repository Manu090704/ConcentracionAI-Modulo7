# Clasificación de imágenes: anime, cartoon y human

Evidencia del módulo de Deep Learning. Una red neuronal convolucional (TensorFlow/Keras) que clasifica una imagen en tres clases: **anime**, **cartoon** o **human**, entrenada con el dataset [Anime vs Cartoon vs Human](https://www.kaggle.com/datasets/hadiepratamatulili/anime-vs-cartoon-vs-human) de Kaggle.

- Autor: Manuel Bajos Rivera (A01711581)

## Estructura del proyecto

```
├── notebooks/
│   └── evidencia.ipynb        # EDA, preprocesado, entrenamiento y evaluación
├── src/
│   ├── preprocesamiento.py    # preprocesado compartido por el notebook y la web
│   └── web/
│       ├── app.py             # servidor Flask: sirve la página y la ruta /predict
│       └── static/            # interfaz: index.html, style.css, script.js
├── imgExternas/               # imágenes que no son del dataset, para probar la generalización
├── modelos/                   # modelo entrenado (.keras) e historial de entrenamiento
├── docs/
│   ├── paper.tex              # reporte (formato IEEE)
│   └── figuras/
├── cache/                     # imágenes preprocesadas (se genera sola, no va en git)
└── requirements.txt
```

## Instalación

Se probó con Python 3.13. Las versiones (TensorFlow 2.20.0 y Keras 3) tienen soporte en macOS, Linux y Windows.

```bash
python -m venv venv
source venv/bin/activate          # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

**Notebook.** Abre `notebooks/evidencia.ipynb` y selecciona el kernel del entorno virtual. El dataset (~4 GB) se descarga la primera vez con `kagglehub`. Las rutas se calculan desde la raíz del repositorio, así que funciona sin importar desde qué carpeta se ejecute.

La primera ejecución genera `cache/`. Si ya existe un modelo en `modelos/`, el notebook lo carga en lugar de volver a entrenar (se puede forzar con `RETRAIN = True`).

**Interfaz web.** Desde la raíz del repositorio:

```bash
python src/web/app.py
```

y abre <http://127.0.0.1:8000>. Se sube una imagen y la página muestra la clase predicha, la probabilidad de cada clase y la imagen tal como la ve el modelo (128×128).

**Reporte.** `docs/paper.tex` usa la clase `ieeeconf`, que no viene incluida y debe estar junto al archivo para compilarlo.

## Datos y preprocesado

- 8,917 imágenes originales, casi balanceadas entre las tres clases.
- Se eliminaron 383 duplicados exactos (todos de anime) antes de partir los datos: quedan 8,534 imágenes.
- Partición estratificada 70/15/15 (entrenamiento, validación y prueba) con semilla 42.
- Cada imagen se convierte a RGB y se redimensiona a 128×128 conservando la proporción, con relleno del color del borde (padding).
- Los píxeles se escalan a [0, 1] dentro del modelo (`Rescaling(1/255)`).

## Resultados

CNN base (3 bloques convolucionales, sin regularización ni aumento de datos):

| Conjunto | Exactitud | F1 macro |
|---|---|---|
| Validación | 99.84% | 0.9985 |
| Prueba | 99.53% | 0.9953 |

**Imágenes externas** (carpeta `imgExternas/`, sección 7 del notebook). Dos de las 10 imágenes son copias de imágenes del dataset (se detectan por contenido y se excluyen), así que quedan 8:

| Conjunto | Exactitud | F1 macro |
|---|---|---|
| Imágenes externas (8) | 25.0% | 0.133 |

Solo acierta las 2 caricaturas: las otras 6 (anime y personas) las clasifica como cartoon con una confianza de ~100%. Quitar el padding no cambia las predicciones. Es un conjunto muy pequeño, pero indica que la exactitud en validación y prueba **no se mantiene con imágenes de otro origen**: el análisis exploratorio mostró que las clases difieren también en resolución, peso de archivo, color y composición, y la validación y la prueba comparten esos sesgos con el entrenamiento.

## Estado

- [x] Descarga de datos, EDA, deduplicación, partición y preprocesado
- [x] CNN base y evaluación en validación y prueba
- [x] Interfaz web
- [x] Prueba con imágenes externas al dataset
- [ ] Mejoras (aumento de datos, regularización, ajuste de hiperparámetros)
- [ ] Transferencia de aprendizaje
- [ ] Reporte final
