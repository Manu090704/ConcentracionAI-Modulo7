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
├── modelos/                   # modelos entrenados (.keras)
├── historiales/               # curvas de entrenamiento de cada modelo (.json), las lee el notebook
├── resultados/                # resultados de las comparaciones con varias semillas (.json), las lee el notebook
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

y abre <http://127.0.0.1:8000>. Un selector permite elegir con cuál de los 9 modelos de `modelos/` se clasifica (por defecto, EfficientNetB0 + aumento de datos con padding, que acierta más imágenes externas). Se sube una imagen y la página muestra la clase predicha, la probabilidad de cada clase y la imagen tal como la ve el modelo (128×128, o 224×224 en el modelo de esa resolución). Al cambiar de modelo con una imagen ya cargada, la predicción se recalcula. Cada modelo recibe la imagen con el mismo redimensionado con que se entrenó (padding o estirar), que el servidor tiene registrado en `src/web/app.py`.

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

**Aumento dirigido a la falla** (sección 11 del notebook). Sobre EfficientNetB0 con imágenes estiradas, se agregaron al aumento anterior tres transformaciones pensadas para simular que el anime no sea solo rostros recortados (hipótesis sin comprobar): alejar la imagen (40-80% del lado, sobre fondo plano), recorte agresivo (30-80% del área) y borrado aleatorio (2-25% del área, con ruido). Se probaron juntas y por separado (ablación):

8. **+ estirar y aumento dirigido**: las tres transformaciones. Una ejecución exploratoria con padding, no guardada, dio 5 de 8 en las externas (F1 0.611), 99.30% en validación y 99.38% en prueba.
9. **+ estirar y alejar**, 10. **+ estirar y recortar** y 11. **+ estirar y borrar**: el aumento de la sección 10 más una sola de las transformaciones (alejar 35%, recortar 35%, borrar 50%).

**Imágenes de 224 × 224** (sección 12 del notebook).

12. **+ estirar y aumento, a 224 px**: el modelo 7 pero con las imágenes transformadas a 224 × 224, la resolución para la que se diseñó EfficientNetB0 (hasta aquí se usaba 128 × 128). La caché va en `cache/stretch_224`.

Cada modelo se evalúa con el mismo redimensionado con que se entrenó. Las cachés de las dos versiones viven en `cache/pad_128` y `cache/stretch_128`.

**Profundidad de la CNN** (sección 13 del notebook). Siguiendo a Gando et al. (2016, [doi:10.1016/j.eswa.2016.08.057](https://doi.org/10.1016/j.eswa.2016.08.057)), que compararon CNN propias de 1 a 5 capas convolucionales, se entrenó la CNN base con 2, 3, 4 y 5 bloques (filtros 32, 64, 128, 256 y 512), 3 semillas cada una (42, 43 y 44). La regla se fijó antes de entrenar: adoptar otra profundidad solo si la validación media no baja más de una desviación estándar respecto a 3 bloques y si acierta en promedio al menos 1 imagen externa más. Los modelos de esta comparación no se guardan; sus resultados están en `resultados/profundidad_cnn.json`.

| Bloques | Parámetros | Validación (media ± DE) | Pérdida de validación | Externas |
|---|---|---|---|---|
| 2 | 7,392,707 | 99.77% ± 0.13 | 0.0059 | 2 de 8 |
| **3 (actual)** | 3,305,027 | **99.84% ± 0.00** | 0.0071 | 2 de 8 |
| 4 | 1,568,579 | 99.51% ± 0.19 | 0.0164 | 2 de 8 |
| 5 | 1,831,235 | 99.45% ± 0.34 | 0.0230 | 2 de 8 |

Se mantienen los 3 bloques: tienen la mejor validación y ninguna profundidad cambia las externas (las 12 redes clasifican las 8 imágenes como cartoon, con al menos 92.9% de confianza). Con la semilla 42, la red de 3 bloques reproduce exactamente la CNN base.

**Profundidad del ajuste fino de EfficientNetB0** (sección 14 del notebook). Las "últimas 20 capas" del ajuste fino también se habían fijado sin fuente. Se siguió el procedimiento incremental de Tajbakhsh et al. (2016, [doi:10.1109/TMI.2016.2535302](https://doi.org/10.1109/TMI.2016.2535302)): empezar por la última capa e ir descongelando hacia atrás, por bloques (solo la cabeza, `top_conv`, `block7a`, `block6a`, … hasta la base completa), y detenerse tras dos pasos seguidos sin mejora. Se aplicó al modelo `stretch_aug` a 128 px con la misma regla y las mismas 3 semillas que la CNN, y la configuración de 20 capas como referencia. Los resultados están en `resultados/profundidad_efficientnet.json`.

| Parte ajustada | Parámetros entrenables | Validación (media ± DE) | Pérdida de validación | Externas (media ± DE) |
|---|---|---|---|---|
| Solo la cabeza | 3,843 | 99.04% ± 0.10 | 0.0251 | 6.00 ± 0.00 de 8 |
| Desde `top_conv` | 413,443 | 99.24% ± 0.07 | 0.0207 | 5.67 ± 0.47 de 8 |
| Desde `block7a` | 1,125,427 | 99.56% ± 0.10 | 0.0127 | 5.00 ± 0.00 de 8 |
| **Últimas 20 capas (actual)** | 1,346,611 | 99.48% ± 0.04 | 0.0136 | 5.00 ± 0.00 de 8 |

El recorrido se detuvo en `block7a`, porque ni `top_conv` ni `block7a` superaron a "solo la cabeza", así que de `block6a` hacia atrás no se entrenó. Se mantienen las 20 capas y no hace falta reentrenar los demás modelos.

Hay un intercambio claro entre los dos criterios:

- **Entrenar solo la cabeza** acierta 6 de 8 con las tres semillas (suma `pruebaAnime.jpg`, con 52–71% de confianza). Pero su validación media (99.04%) queda por debajo del piso de la regla (99.44%, unas 6 imágenes más mal clasificadas de 1,281).
- **Ajustar desde `block7a`** tiene la mejor validación, pero no cambia las externas.

Una posible explicación, sin comprobar, es que ajustar más capas adapta la red a los sesgos del dataset. La diferencia depende de 1 imagen de 8, así que no es concluyente.

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
| EfficientNetB0 + estirar y aumento dirigido | 99.45% / 0.9945 | 99.30% / 0.9929 | 9 | 50.0% / 0.490 |
| EfficientNetB0 + estirar y alejar | 99.14% / 0.9914 | 99.45% / 0.9945 | 7 | 50.0% / 0.489 |
| EfficientNetB0 + estirar y recortar | 99.61% / 0.9961 | 99.69% / 0.9969 | 4 | 62.5% / 0.635 |
| EfficientNetB0 + estirar y borrar | 99.61% / 0.9961 | 99.77% / 0.9976 | 3 | 62.5% / 0.611 |
| EfficientNetB0 + estirar y aumento (224 px) | 99.84% / 0.9985 | 99.92% / 0.9992 | 1 | 62.5% / 0.635 |

En validación y prueba todos los modelos superan el 99.5% y se distinguen por una a tres imágenes, así que esos conjuntos ya no sirven para compararlos.

**Imágenes externas** (carpeta `imgExternas/`, sección 7 del notebook). Son 8 imágenes; el notebook las compara por contenido (MD5) con el dataset y excluiría cualquier copia. Se obtuvieron de forma aleatoria de internet, sin verificar su licencia ni registrar su fuente; se incluyen solo con fines académicos, para evaluar los modelos, y sus derechos pertenecen a sus autores.

- La CNN base solo acierta las 2 caricaturas: las otras 6 (anime y personas) las clasifica como cartoon con una confianza de ~100%. Quitar el padding no cambia las predicciones.
- EfficientNetB0 acierta 4 de 8 (las 2 caricaturas, un rostro anime pixelado y una foto de persona). Sigue fallando con ilustraciones anime modernas, una viñeta de manga y un modelo anatómico, y todos sus errores van a cartoon.

- **Dropout no cambia ninguna predicción externa** en ninguna de las dos arquitecturas (mismos aciertos y mismos errores). Sí cambia las curvas de la CNN (su pérdida de entrenamiento final es 100 veces mayor y las curvas son más irregulares), pero no mejora la validación. Es coherente con que el problema no es un sobreajuste clásico: la exactitud de entrenamiento y la de validación ya eran casi iguales, y el error está en que las imágenes externas son distintas a las del dataset.

- **Estirar** no cambia ningún acierto externo, pero baja la seguridad de los errores (una ilustración anime pasa de 100% a 84% de confianza como cartoon).
- **El aumento de datos** corrige una imagen más (la viñeta de manga en blanco y negro), con padding o estirando: 5 de 8. Combinar ambos cambios no mejora respecto al aumento solo.
- En validación y prueba las variantes con cambios en los datos rinden un poco peor (99.69% frente a 99.92% en validación), el costo esperable de entrenar con datos más variados.
- **El aumento dirigido no mejoró** (4 de 8, con 9 errores en prueba frente a 3). Por separado: **alejar** es la única transformación que acierta una imagen nueva (`pruebaAnime.jpg`), pero pierde la viñeta de manga y el rostro pixelado y es la que más empeora validación y prueba; **recortar** y **borrar** aciertan las mismas imágenes que el modelo base, sin cambio. Cada variante se entrenó una sola vez y con 8 imágenes una diferencia de una imagen no se distingue del azar.
- **A 224 px** la exactitud sube en validación (99.84%) y en prueba (99.92%, 1 error, la mejor de las variantes estiradas), pero las externas se quedan en 5 de 8: `pruebaAnime2.jpg` se acierta por primera vez (anime, 65%) y se pierde el rostro pixelado `pruebaAnime4.jpg` (cartoon, 98%). Un solo entrenamiento y 8 imágenes: no concluyente.
- **Ninguna variante acierta a la vez las dos ilustraciones anime de estilo moderno**, el modelo anatómico falla en todas y casi todos los errores van a cartoon. Una posible explicación (sin comprobar) es que el anime del dataset son solo rostros recortados, y el aumento no crea personajes con cuerpo, fondos ni estilos nuevos.

Es un conjunto muy pequeño y las diferencias de 1 o 2 aciertos (transfer learning, aumento de datos) o de 1 imagen en prueba (Dropout) no son concluyentes, pero indica que la exactitud en validación y prueba **no se mantiene con imágenes de otro origen**. El análisis exploratorio mostró que las clases difieren también en resolución, peso de archivo, color y composición, y la validación y la prueba comparten esos sesgos con el entrenamiento. Por eso las mejoras siguientes se comparan con las imágenes externas y no solo con la validación, que ya está saturada.

## Estado

- [x] Descarga de datos, EDA, deduplicación, partición y preprocesado
- [x] CNN base y evaluación en validación y prueba
- [x] Interfaz web
- [x] Prueba con imágenes externas al dataset
- [x] Transferencia de aprendizaje (EfficientNetB0)
- [x] Regularización con Dropout (sin mejora medible)
- [x] Aumento de datos y estirar las imágenes (mejora modesta)
- [x] Profundidad de la CNN, de 2 a 5 bloques (sin mejora: se mantienen 3)
- [x] Profundidad del ajuste fino de EfficientNetB0 (sin mejora que cumpla la regla: se mantienen 20 capas)
- [ ] Reporte final
