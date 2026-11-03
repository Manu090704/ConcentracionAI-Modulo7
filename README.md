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

La primera ejecución genera `cache/`. Si ya existe un modelo en `modelos/`, el notebook lo carga en lugar de volver a entrenar (se puede forzar con `RETRAIN`, `RETRAIN_TL`, `RETRAIN_CNN_DO` o `RETRAIN_TL_DO`, según el modelo).

**Interfaz web.** Desde la raíz del repositorio:

```bash
python src/web/app.py
```

y abre <http://127.0.0.1:8000>. Un selector permite elegir con cuál de los 7 modelos de `modelos/` se clasifica (por defecto, EfficientNetB0 + aumento de datos con padding, que acierta más imágenes externas). Se sube una imagen y la página muestra la clase predicha, la probabilidad de cada clase y la imagen tal como la ve el modelo (128×128). Al cambiar de modelo con una imagen ya cargada, la predicción se recalcula. Cada modelo recibe la imagen con el mismo redimensionado con que se entrenó (padding o estirar), que el servidor tiene registrado en `src/web/app.py`.

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
3. **CNN + Dropout**: la CNN base con `Dropout(0.5)` después de la capa densa de 128 neuronas. Todo lo demás igual.
4. **EfficientNetB0 + Dropout**: EfficientNetB0 con `Dropout(0.2)` después del `GlobalAveragePooling2D` (la tasa de su clasificador original). Todo lo demás igual.

Cada tasa de Dropout se fijó de antemano con un valor estándar y no se ajustó, para no sobreajustar a un conjunto externo tan pequeño. Cada modelo se entrenó una sola vez.

**Tratamiento de los datos** (sección 10 del notebook). Sobre EfficientNetB0 sin Dropout, con el mismo entrenamiento en dos fases, se probaron dos cambios por separado y juntos:

5. **+ estirar**: las imágenes se estiran a 128×128 en lugar de conservar la proporción con padding (elimina las bandas de relleno, que solo aparecían en las cartoon).
6. **+ aumento de datos (padding)**: transformaciones aleatorias solo en entrenamiento: volteo, rotación (hasta 11°), zoom de hasta 25% con proporción variable, traslación, brillo/contraste/saturación, escala de grises (15%) y baja resolución (50%).
7. **+ estirar y aumento de datos**: ambos cambios.

Cada modelo se evalúa con el mismo redimensionado con que se entrenó. Las cachés de las dos versiones viven en `cache/pad_128` y `cache/stretch_128`.

## Resultados

| Modelo | Validación (exactitud / F1) | Prueba (exactitud / F1) | Errores en prueba | Externas, 8 imágenes (exactitud / F1) |
|---|---|---|---|---|
| CNN base | 99.84% / 0.9985 | 99.53% / 0.9953 | 6 de 1,281 | 25.0% / 0.133 |
| CNN + Dropout | 99.84% / 0.9985 | 99.61% / 0.9960 | 5 | 25.0% / 0.133 |
| EfficientNetB0 | 99.92% / 0.9992 | 99.92% / 0.9992 | 1 | 50.0% / 0.522 |
| EfficientNetB0 + Dropout | 99.92% / 0.9992 | 100% / 1.0000 | 0 | 50.0% / 0.522 |
| EfficientNetB0 + estirar | 99.69% / 0.9968 | 99.84% / 0.9984 | 2 | 50.0% / 0.522 |
| EfficientNetB0 + aumento (padding) | 99.69% / 0.9969 | 99.84% / 0.9984 | 2 | 62.5% / 0.635 |
| EfficientNetB0 + estirar y aumento | 99.69% / 0.9968 | 99.77% / 0.9976 | 3 | 62.5% / 0.635 |

En validación y prueba todos los modelos superan el 99.5% y se distinguen por una a tres imágenes, así que esos conjuntos ya no sirven para compararlos.

**Imágenes externas** (carpeta `imgExternas/`, sección 7 del notebook). Son 8 imágenes; el notebook las compara por contenido (MD5) con el dataset y excluiría cualquier copia. Se obtuvieron de forma aleatoria de internet, sin verificar su licencia ni registrar su fuente; se incluyen solo con fines académicos, para evaluar los modelos, y sus derechos pertenecen a sus autores.

- La CNN base solo acierta las 2 caricaturas: las otras 6 (anime y personas) las clasifica como cartoon con una confianza de ~100%. Quitar el padding no cambia las predicciones.
- EfficientNetB0 acierta 4 de 8 (las 2 caricaturas, un rostro anime pixelado y una foto de persona). Sigue fallando con ilustraciones anime modernas, una viñeta de manga y un modelo anatómico, y todos sus errores van a cartoon.

- **Dropout no cambia ninguna predicción externa** en ninguna de las dos arquitecturas (mismos aciertos y mismos errores). Sí cambia las curvas de la CNN (su pérdida de entrenamiento final es 100 veces mayor y las curvas son más irregulares), pero no mejora la validación. Es coherente con que el problema no es un sobreajuste clásico: la exactitud de entrenamiento y la de validación ya eran casi iguales, y el error está en que las imágenes externas son distintas a las del dataset.

- **Estirar** no cambia ningún acierto externo, pero baja la seguridad de los errores (una ilustración anime pasa de 100% a 84% de confianza como cartoon).
- **El aumento de datos** corrige una imagen más (la viñeta de manga en blanco y negro), con padding o estirando: 5 de 8. Combinar ambos cambios no mejora respecto al aumento solo.
- En validación y prueba las variantes con cambios en los datos rinden un poco peor (99.69% frente a 99.92% en validación), el costo esperable de entrenar con datos más variados.
- **En todas las variantes siguen fallando** las ilustraciones anime de estilo moderno y el modelo anatómico, y todos los errores van a cartoon. Una posible explicación (sin comprobar) es que el anime del dataset son solo rostros recortados, y el aumento no crea personajes con cuerpo, fondos ni estilos nuevos.

Es un conjunto muy pequeño y las diferencias de 1 o 2 aciertos (transfer learning, aumento de datos) o de 1 imagen en prueba (Dropout) no son concluyentes, pero indica que la exactitud en validación y prueba **no se mantiene con imágenes de otro origen**. El análisis exploratorio mostró que las clases difieren también en resolución, peso de archivo, color y composición, y la validación y la prueba comparten esos sesgos con el entrenamiento. Por eso las mejoras siguientes se comparan con las imágenes externas y no solo con la validación, que ya está saturada.

## Estado

- [x] Descarga de datos, EDA, deduplicación, partición y preprocesado
- [x] CNN base y evaluación en validación y prueba
- [x] Interfaz web
- [x] Prueba con imágenes externas al dataset
- [x] Transferencia de aprendizaje (EfficientNetB0)
- [x] Regularización con Dropout (sin mejora medible)
- [x] Aumento de datos y estirar las imágenes (mejora modesta)
- [ ] Reporte final
