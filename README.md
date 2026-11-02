# Clasificación de imágenes: anime, cartoon y human

Evidencia del módulo de Deep Learning. Redes neuronales convolucionales (TensorFlow/Keras) que clasifican una imagen en tres clases: **anime**, **cartoon** o **human**, entrenadas con el dataset [Anime vs Cartoon vs Human](https://www.kaggle.com/datasets/hadiepratamatulili/anime-vs-cartoon-vs-human) de Kaggle.

- Autor: Manuel Bajos Rivera (A01711581)

## Estructura del proyecto

```
├── notebooks/
│   └── evidencia.ipynb        # EDA, preprocesado, entrenamiento (CNN base y transfer learning) y evaluación
├── src/
│   ├── preprocesamiento.py    # preprocesado compartido por el notebook y la web
│   └── web/
│       ├── app.py             # servidor Flask: sirve la página y la ruta /predict
│       └── static/            # interfaz: index.html, style.css, script.js
├── imgExternas/               # imágenes que no son del dataset, para probar la generalización
├── modelos/                   # modelos entrenados (.keras) e historial de entrenamiento
├── docs/
│   ├── paper.tex              # reporte (formato IEEE)
│   └── figuras/               # figuras del reporte (varias las genera el notebook)
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

**Pesos preentrenados.** El modelo con transfer learning descarga los pesos de ImageNet la primera vez que se construye. En macOS con Python instalado desde python.org puede fallar con un error de certificados SSL (`CERTIFICATE_VERIFY_FAILED`); se resuelve ejecutando una vez el archivo `Install Certificates.command` de la carpeta de Python en `/Applications/`.

## Uso

**Notebook.** Abre `notebooks/evidencia.ipynb` y selecciona el kernel del entorno virtual. El dataset (~4 GB) se descarga la primera vez con `kagglehub`. Las rutas se calculan desde la raíz del repositorio, así que funciona sin importar desde qué carpeta se ejecute.

La primera ejecución genera `cache/`. Si ya existe un modelo en `modelos/`, el notebook lo carga en lugar de volver a entrenar (se puede forzar con `RETRAIN = True` para la CNN base y `RETRAIN_TL = True` para EfficientNetB0).

**Interfaz web.** Desde la raíz del repositorio:

```bash
python src/web/app.py
```

y abre <http://127.0.0.1:8000>. Se sube una imagen y la página muestra la clase predicha, la probabilidad de cada clase y la imagen tal como la ve el modelo (128×128). Por ahora la interfaz usa la CNN base (`modelos/cnn_base_pad.keras`).

**Reporte.** `docs/paper.tex` usa la clase `ieeeconf`, que no viene incluida y debe estar junto al archivo para compilarlo.

## Datos y preprocesado

- 8,917 imágenes originales, casi balanceadas entre las tres clases.
- Se eliminaron 383 duplicados exactos (todos de anime) antes de partir los datos: quedan 8,534 imágenes.
- Partición estratificada 70/15/15 (entrenamiento, validación y prueba) con semilla 42.
- Cada imagen se convierte a RGB y se redimensiona a 128×128 conservando la proporción, con relleno del color del borde (padding).
- La CNN base escala los píxeles a [0, 1] dentro del modelo (`Rescaling(1/255)`). EfficientNetB0 incluye su propio preprocesado y espera píxeles en [0, 255], así que no se divide entre 255.

## Modelos

1. **CNN base** (desde cero): tres bloques convolucionales (32, 64 y 128 filtros) y un clasificador denso, 3.3 millones de parámetros. Sin regularización ni aumento de datos. Adam (lr 0.001), 15 épocas.
2. **EfficientNetB0 con transfer learning**: pesos de ImageNet y una cabeza nueva (`GlobalAveragePooling2D` y `Dense(3)`). Fase 1: base congelada, solo la cabeza (8 épocas, lr 0.001). Fase 2: ajuste fino de las últimas 20 capas de la base, sin tocar las `BatchNormalization` (10 épocas, lr 0.00001). Mismos datos y particiones, sin aumento de datos ni regularización adicional.

## Resultados

| Modelo | Validación (exactitud / F1) | Prueba (exactitud / F1) | Externas, 8 imágenes (exactitud / F1) |
|---|---|---|---|
| CNN base | 99.84% / 0.9985 | 99.53% / 0.9953 | 25.0% / 0.133 |
| EfficientNetB0 | 99.92% / 0.9992 | 99.92% / 0.9992 | 50.0% / 0.522 |

En prueba la CNN base comete 6 errores de 1,281 imágenes y EfficientNetB0 uno.

**Imágenes externas** (carpeta `imgExternas/`, sección 7 del notebook). Dos de las 10 imágenes son copias de imágenes del dataset (se detectan por contenido y se excluyen), así que quedan 8.

- La CNN base solo acierta las 2 caricaturas: las otras 6 (anime y personas) las clasifica como cartoon con una confianza de ~100%. Quitar el padding no cambia las predicciones.
- EfficientNetB0 acierta 4 de 8 (las 2 caricaturas, un rostro anime pixelado y una foto de persona). Sigue fallando con ilustraciones anime modernas, una viñeta de manga y un modelo anatómico, y todos sus errores van a cartoon.

Es un conjunto muy pequeño y la diferencia de 2 aciertos no es concluyente, pero indica que la exactitud en validación y prueba **no se mantiene con imágenes de otro origen**. El análisis exploratorio mostró que las clases difieren también en resolución, peso de archivo, color y composición, y la validación y la prueba comparten esos sesgos con el entrenamiento. Por eso las mejoras siguientes se comparan con las imágenes externas y no solo con la validación, que ya está saturada.

## Estado

- [x] Descarga de datos, EDA, deduplicación, partición y preprocesado
- [x] CNN base y evaluación en validación y prueba
- [x] Interfaz web
- [x] Prueba con imágenes externas al dataset
- [x] Transferencia de aprendizaje (EfficientNetB0)
- [ ] Regularización (Dropout, L2, early stopping)
- [ ] Aumento de datos
- [ ] Reporte final
