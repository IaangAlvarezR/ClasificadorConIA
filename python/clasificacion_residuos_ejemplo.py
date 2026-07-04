"""
Ejemplo base para clasificar residuos reciclables y no reciclables.

Dataset sugerido:
https://www.kaggle.com/datasets/ashwinshrivastav/most-common-recyclable-and-nonrecyclable-objects

Uso recomendado:
1. Descarga el dataset desde Kaggle y deja las imagenes organizadas por carpetas:

   dataset/
     recyclable/
       imagen_1.jpg
       imagen_2.jpg
     non_recyclable/
       imagen_3.jpg
       imagen_4.jpg

   Tambien funciona si el dataset tiene mas clases, por ejemplo plastic, paper,
   glass, organic, etc. El script aprende una clase por carpeta.

2. Instala dependencias:

   pip install tensorflow fastapi uvicorn python-multipart pillow numpy

3. Entrena:

   python clasificacion_residuos_ejemplo.py train --data ./dataset --epochs 10

4. Prueba una imagen:

   python clasificacion_residuos_ejemplo.py predict --image ./prueba.jpg

5. Levanta backend para Railway:

   uvicorn clasificacion_residuos_ejemplo:app --host 0.0.0.0 --port 8000

El frontend actual puede consumir este backend porque expone POST /predict.
En Vercel configura VITE_API_URL con la URL publica de Railway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Obtener el directorio actual donde se encuentra este archivo de Python
BASE_DIR = Path(__file__).resolve().parent

# Configurar las rutas absolutas apuntando directamente al archivo en la misma carpeta
MODEL_PATH = os.path.join(BASE_DIR, "modelo_residuos.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

app = FastAPI(title="EcoClasifica IA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NUEVA LÓGICA DE CARGA GLOBAL ---
MODELO_GLOBAL = None
ETIQUETAS_GLOBAL = None

@app.on_event("startup")
def load_model_on_startup():
    """Carga el modelo de IA una sola vez cuando el servidor se enciende"""
    global MODELO_GLOBAL, ETIQUETAS_GLOBAL
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        print("Cargando modelo de IA en memoria...")
        MODELO_GLOBAL = tf.keras.models.load_model(MODEL_PATH)
        ETIQUETAS_GLOBAL = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        print("¡Modelo cargado exitosamente!")
    else:
        print("Advertencia: No se encontraron los archivos del modelo en la ruta especificada.")


def build_model(num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(data_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def train(data_dir: Path, epochs: int) -> None:
    train_ds, val_ds, class_names = load_datasets(data_dir)
    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)

    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Modelo guardado en: {MODEL_PATH.resolve()}")
    print(f"Etiquetas guardadas en: {LABELS_PATH.resolve()}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def predict_from_image(image: Image.Image) -> dict[str, Any]:
    global MODELO_GLOBAL, ETIQUETAS_GLOBAL
    
    if MODELO_GLOBAL is None or ETIQUETAS_GLOBAL is None:
        raise FileNotFoundError("El modelo no está cargado en el servidor.")
        
    batch = preprocess_image(image)
    probabilities = MODELO_GLOBAL.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    return {
        "label": ETIQUETAS_GLOBAL[best_index],
        "confidence": round(float(probabilities[best_index]) * 100, 2),
        "probabilities": {
            ETIQUETAS_GLOBAL[index]: round(float(value) * 100, 2)
            for index, value in enumerate(probabilities)
        },
    }


def predict_file(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    image = Image.open(image_path)
    result = predict_from_image(image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        return predict_from_image(image)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        print(f"Error interno al procesar: {error}")
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")


def parse_args() -> argparse.Namespace | None:
    import sys
    if len(sys.argv) == 1 or "uvicorn" in sys.argv[0]:
        return None
        
    parser = argparse.ArgumentParser(description="Clasificador de residuos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el modelo")
    train_parser.add_argument("--data", type=Path, required=True, help="Carpeta del dataset")
    train_parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")

    predict_parser = subparsers.add_parser("predict", help="Predice una imagen")
    predict_parser.add_argument("--image", type=Path, required=True, help="Ruta de la imagen")

    return parser.parse_args()


# ... (Todo el resto del código optimizado que te pasé arriba se mantiene exactamente igual)

if __name__ == "__main__":
    args = parse_args()

    # Si NO se pasaron argumentos de consola (es decir, lo está ejecutando Railway)
    if args is None:
        import uvicorn
        import os
        # Lee el puerto dinámico de Railway, y si no existe usa el 8000 por defecto
        puerto = int(os.environ.get("PORT", 8000))
        print(f"Iniciando servidor de producción en el puerto {puerto}...")
        uvicorn.run("python.clasificacion_residuos_ejemplo:app", host="0.0.0.0", port=puerto)
    else:
        if args.command == "train":
            train(args.data, args.epochs)
        elif args.command == "predict":
            predict_file(args.image)