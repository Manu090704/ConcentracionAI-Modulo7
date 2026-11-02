"""Preprocesado de imagenes, compartido por el notebook de entrenamiento y la interfaz web.

Tener una sola implementacion garantiza que el modelo recibe en produccion exactamente
lo mismo que vio al entrenar.
"""
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 128
CLASSES = ["anime", "cartoon", "human"]  # mismo orden que en el entrenamiento (carpetas en orden alfabetico)


def to_rgb(im):
    """Pasa a RGB; si hay transparencia, la pone sobre fondo blanco."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return im.convert("RGB")


def edge_color(arr):
    """Color mediano del borde de la imagen, para rellenar sin crear barras llamativas."""
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return np.median(border, axis=0).astype(np.uint8)


def preprocess(im, size=IMG_SIZE, mode="pad"):
    """Imagen PIL -> arreglo uint8 de size x size x 3.

    mode="pad": conserva la proporcion y rellena con el color del borde.
    mode="stretch": estira la imagen a un cuadrado.
    """
    im = to_rgb(im)
    if mode == "stretch":
        return np.asarray(im.resize((size, size), Image.Resampling.LANCZOS))
    scale = size / max(im.size)
    new_w, new_h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    small = np.asarray(im.resize((new_w, new_h), Image.Resampling.LANCZOS))
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = edge_color(small)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small
    return canvas
