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

5. Levanta backend para Railway/Render:

   uvicorn clasificacion_residuos_ejemplo:app --host 0.0.0.0 --port 8000

El frontend actual puede consumir este backend porque expone POST /predict.
En Vercel configura VITE_API_URL con la URL publica de tu servidor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MIN_RECYCLING_CONFIDENCE = float(os.environ.get("MIN_RECYCLING_CONFIDENCE", "0.76"))
MIN_RECYCLING_MARGIN = float(os.environ.get("MIN_RECYCLING_MARGIN", "0.28"))
UNRELATED_IMAGE_DETAIL = (
    "La imagen no parece corresponder a un residuo reconocible. "
    "Sube una foto clara de basura, envases, botellas, latas, carton, plastico o materiales reciclables."
)
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB máximo por imagen
ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

# Obtener el directorio actual donde se encuentra este archivo de Python
BASE_DIR = Path(__file__).resolve().parent


def load_dotenv(env_path: Path | str | None = None) -> None:
    """Carga variables de entorno desde un archivo .env local si existe."""
    env_path = Path(env_path or BASE_DIR / ".env")
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue

            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and os.environ.get(key) is None:
                os.environ[key] = value


load_dotenv()

# Configurar las rutas absolutas apuntando directamente al archivo en la misma carpeta como cadenas de texto (str)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_residuos.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

app = FastAPI(title="EcoClasifica IA API")


class ChatRequest(BaseModel):
    question: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LÓGICA DE CARGA GLOBAL DEL MODELO ---
MODELO_GLOBAL = None
ETIQUETAS_GLOBAL = None
CHAT_REPLY_CACHE: dict[str, str] = {}
LLM_DISABLED_UNTIL = 0.0

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
LLM_RETRY_COUNT = int(os.environ.get("LLM_RETRY_COUNT", "2"))
LLM_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "1.5"))
LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "12"))
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "96"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-flash-lite-latest")
GOOGLE_FALLBACK_MODELS = [
    model.strip()
    for model in os.environ.get(
        "GOOGLE_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-flash-lite-latest",
    ).split(",")
    if model.strip()
]
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

@app.on_event("startup")
def load_model_on_startup():
    """Carga el modelo de IA una sola vez cuando el servidor se enciende"""
    global MODELO_GLOBAL, ETIQUETAS_GLOBAL

    if tf is None:
        print("Advertencia: TensorFlow no está instalado; el chat seguirá funcionando, pero /predict no estará disponible.")
        return

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        print("Cargando modelo de IA en memoria...")
        MODELO_GLOBAL = tf.keras.models.load_model(MODEL_PATH)

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            ETIQUETAS_GLOBAL = json.load(f)

        print("¡Modelo cargado exitosamente!")
    else:
        print("Advertencia: No se encontraron los archivos del modelo en la ruta especificada.")


def build_model(num_classes: int) -> Any:
    if tf is None:
        raise RuntimeError("TensorFlow no está instalado. Instala una versión compatible con Python 3.11/3.12/3.13 para usar /predict.")

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


def load_datasets(data_dir: Path) -> tuple[Any, Any, list[str]]:
    if tf is None:
        raise RuntimeError("TensorFlow no está instalado. Instala una versión compatible con Python 3.11/3.12/3.13 para entrenar o usar /predict.")

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

    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
        
    print(f"Modelo guardado en: {os.path.abspath(MODEL_PATH)}")
    print(f"Etiquetas guardadas en: {os.path.abspath(LABELS_PATH)}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    if tf is None:
        raise RuntimeError("TensorFlow no está instalado. Instala una versión compatible con Python 3.11/3.12/3.13 para usar /predict.")

    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def validate_prediction_confidence(probabilities: np.ndarray) -> None:
    scores = np.asarray(probabilities, dtype=float).ravel()

    if scores.size == 0:
        raise HTTPException(status_code=422, detail=UNRELATED_IMAGE_DETAIL)

    sorted_scores = np.sort(scores)
    best_score = float(sorted_scores[-1])
    margin = best_score if sorted_scores.size == 1 else best_score - float(sorted_scores[-2])

    if best_score < MIN_RECYCLING_CONFIDENCE or margin < MIN_RECYCLING_MARGIN:
        raise HTTPException(status_code=422, detail=UNRELATED_IMAGE_DETAIL)


def predict_from_image(image: Image.Image) -> dict[str, Any]:
    global MODELO_GLOBAL, ETIQUETAS_GLOBAL

    if tf is None:
        raise RuntimeError("TensorFlow no está instalado. Instala una versión compatible con Python 3.11/3.12/3.13 para usar /predict.")

    if MODELO_GLOBAL is None or ETIQUETAS_GLOBAL is None:
        raise FileNotFoundError("El modelo no está cargado en el servidor.")

    batch = preprocess_image(image)
    probabilities = MODELO_GLOBAL.predict(batch, verbose=0)[0]
    validate_prediction_confidence(probabilities)
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


def build_local_reply(question: str) -> str:
    normalized = unicodedata.normalize("NFKD", question)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^\w\s]", "", normalized).lower()

    if any(token in normalized for token in ["hola", "ayuda", "qué puedes hacer", "como funciona"]):
        return "Puedo responder preguntas sobre reciclaje, clasificación de residuos y las categorías que usa este proyecto: reciclable y no reciclable."

    if any(token in normalized for token in ["reciclable", "reciclaje", "reciclar", "papel", "carton", "vidrio", "metal", "botella", "lata", "envase"]):
        return "En general, los materiales como vidrio, latas, botellas limpias y papel o cartón secos suelen ser reciclables, siempre que estén limpios y separados correctamente."

    if any(token in normalized for token in ["tetrapack", "caja", "jugo", "leche", "envase carton"]):
        return "Los envases tipo tetrapack o cajas de jugo suelen requerir atención especial. Muchas veces se reciclan, pero dependen del tipo de material y de la infraestructura local."

    if any(token in normalized for token in ["styrofoam", "espuma", "poliestireno"]):
        return "El styrofoam o espuma de poliestireno suele considerarse difícil de reciclar y, en muchos casos, se clasifica como no reciclable para este proyecto."

    if any(token in normalized for token in ["utensilio", "cuchara", "tenedor", "cubiertos", "plástico de un solo uso"]):
        return "Los utensilios y los plásticos de un solo uso suelen entrar en la categoría de no reciclable cuando están muy contaminados o cuando no son aceptados por la red de reciclaje local."

    if any(token in normalized for token in ["ia", "modelo", "cnn", "clasificador", "proyecto", "clase", "categoría"]):
        return "Este proyecto usa una IA para clasificar residuos en dos grandes categorías: reciclable y no reciclable, con ejemplos como botellas, latas, cajas, utensilios y empaques especiales."

    return "Solo puedo ayudar con preguntas relacionadas con reciclaje, residuos y las clases que se manejan en este proyecto. Si quieres, pregúntame por botellas, latas, tetrapack, utensilios o styrofoam."


def build_local_reply(question: str) -> str:
    normalized = unicodedata.normalize("NFKD", question)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^\w\s]", "", normalized).lower()

    local_replies = [
        (
            ["hola", "ayuda", "que puedes hacer", "como funciona"],
            "Puedo ayudarte con dudas sobre reciclaje, separacion de residuos, limpieza de envases y las categorias del proyecto: reciclable y no reciclable.",
        ),
        (
            ["limpiar", "lavar", "sucio", "grasoso", "comida", "contaminado"],
            "Si un envase tiene restos de comida, aceite o liquidos, conviene vaciarlo y enjuagarlo. Un residuo limpio tiene mas probabilidades de ser reciclable.",
        ),
        (
            ["papel", "carton", "periodico", "cuaderno", "caja de carton"],
            "Papel y carton suelen ser reciclables si estan secos y limpios. Si estan mojados, con grasa o con restos de comida, pueden dejar de aceptarse.",
        ),
        (
            ["vidrio", "frasco", "botella de vidrio"],
            "El vidrio limpio, como botellas y frascos, suele ser reciclable. Evita mezclarlo con ceramica, espejos o focos, porque normalmente requieren manejo especial.",
        ),
        (
            ["metal", "aluminio", "lata", "conserva"],
            "Las latas de aluminio o metal limpias suelen ser reciclables. Lo mejor es vaciarlas, enjuagarlas y separarlas de residuos organicos.",
        ),
        (
            ["plastico", "pet", "bolsa", "botella plastica", "envase plastico"],
            "Muchos plasticos, como botellas PET limpias, pueden reciclarse. Bolsas, envolturas flexibles y plasticos sucios dependen mucho del centro de acopio local.",
        ),
        (
            ["tetrapack", "tetrabrik", "jugo", "leche", "caja"],
            "Los envases tipo tetrapack pueden reciclarse en lugares que aceptan materiales multicapa. Vacialos, enjuagalos y aplastalos antes de separarlos.",
        ),
        (
            ["organico", "comida", "fruta", "verdura", "cascaras", "jardin"],
            "Los residuos organicos, como restos de fruta, verdura y poda, no van con reciclables secos. Pueden aprovecharse en composta si estan separados.",
        ),
        (
            ["pila", "bateria", "electronico", "celular", "cargador", "cable"],
            "Pilas, baterias y electronicos no deben tirarse con basura comun. Llevan metales y componentes que requieren puntos de recoleccion especializados.",
        ),
        (
            ["aceite", "cocina", "fritura"],
            "El aceite usado no debe ir al drenaje. Guardalo frio en una botella cerrada y llevalo a un punto de recoleccion si existe en tu localidad.",
        ),
        (
            ["medicina", "medicamento", "pastilla", "jarabe"],
            "Los medicamentos caducos no deben mezclarse con reciclables ni tirarse al drenaje. Lo ideal es llevarlos a un contenedor o farmacia con programa de acopio.",
        ),
        (
            ["ropa", "textil", "zapato", "tela"],
            "La ropa y textiles no suelen clasificarse como reciclables comunes. Si estan en buen estado, donarlos o reutilizarlos suele ser mejor opcion.",
        ),
        (
            ["unicel", "styrofoam", "espuma", "poliestireno"],
            "El unicel o espuma de poliestireno suele ser dificil de reciclar y muchas redes no lo aceptan, especialmente si tiene restos de comida.",
        ),
        (
            ["contenedor", "color", "verde", "amarillo", "azul", "separar"],
            "Los colores de contenedores cambian por ciudad, pero la regla practica es separar reciclables limpios y secos, organicos, residuos sanitarios y residuos especiales.",
        ),
        (
            ["sanitario", "papel higienico", "panal", "toalla femenina", "cubrebocas"],
            "Los residuos sanitarios no se consideran reciclables. Deben ir cerrados en una bolsa y separados de materiales limpios como papel, carton, vidrio o metal.",
        ),
        (
            ["foco", "lampara", "bombilla", "fluorescente"],
            "Focos, lamparas y fluorescentes requieren manejo especial. No los mezcles con vidrio comun porque pueden contener componentes peligrosos.",
        ),
        (
            ["ia", "modelo", "cnn", "clasificador", "proyecto", "clase", "categoria"],
            "El proyecto usa un modelo de IA para distinguir residuos reciclables y no reciclables. Si la imagen no parece un residuo, el sistema puede rechazarla por baja confianza.",
        ),
    ]

    for tokens, reply in local_replies:
        if any(token in normalized for token in tokens):
            return reply

    return "Puedo orientarte sobre residuos reciclables, no reciclables y manejo responsable. Dame el material o ejemplo concreto, como botella, lata, carton, pila, aceite, ropa o tetrapack."


def get_openai_reply(question: str) -> str | None:
    if not OPENAI_API_KEY:
        return None

    try:
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Responde únicamente sobre reciclaje, residuos y las clases del proyecto EcoClasifica IA. Nunca respondas fuera de ese contexto.",
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
            "max_tokens": LLM_MAX_OUTPUT_TOKENS,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None
    except urllib.error.HTTPError as error:
        if error.code in {429, 500, 502, 503, 504}:
            return "__RATE_LIMIT__"
        print(f"Error al consultar OpenAI: {error}")
        return None
    except Exception as error:
        print(f"Error al consultar OpenAI: {error}")
        return None


def get_gemini_reply(question: str) -> str | None:
    if not GOOGLE_API_KEY:
        return None

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": question}],
            }
        ],
        "systemInstruction": {
            "parts": [
                    {
                        "text": (
                            "Responde unicamente sobre reciclaje, residuos y las clases del proyecto "
                            "EcoClasifica IA. Nunca respondas fuera de ese contexto. Responde en espanol, "
                            "en 1 a 3 frases cortas. Se directo, sin listas largas ni explicaciones extensas."
                        )
                    }
                ]
            },
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": LLM_MAX_OUTPUT_TOKENS,
                "candidateCount": 1,
            },
    }
    models_to_try = list(dict.fromkeys([GOOGLE_MODEL, *GOOGLE_FALLBACK_MODELS]))
    had_temporary_error = False

    for attempt in range(LLM_RETRY_COUNT + 1):
        for model in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": GOOGLE_API_KEY,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=LLM_REQUEST_TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    candidates = body.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(part.get("text", "") for part in parts)
                        return text.strip() or None
            except urllib.error.HTTPError as error:
                if error.code in {429, 500, 502, 503, 504}:
                    had_temporary_error = True
                    continue
                print(f"Error al consultar Gemini: {error}")
                return None
            except Exception as error:
                had_temporary_error = True
                print(f"Error al consultar Gemini: {error}")

        if attempt < LLM_RETRY_COUNT:
            time.sleep(LLM_RETRY_DELAY)

    if had_temporary_error:
        return "__RATE_LIMIT__"
    return None


def get_chat_reply(question: str) -> str:
    global LLM_DISABLED_UNTIL

    normalized_question = question.strip().lower()
    if normalized_question in CHAT_REPLY_CACHE:
        return CHAT_REPLY_CACHE[normalized_question]

    if time.time() < LLM_DISABLED_UNTIL:
        return build_local_reply(question)

    reply = None
    if LLM_PROVIDER == "google":
        reply = get_gemini_reply(question)
    else:
        reply = get_openai_reply(question)

    if reply == "__RATE_LIMIT__":
        LLM_DISABLED_UNTIL = time.time() + 60
        return build_local_reply(question)
    elif reply is None:
        return build_local_reply(question)

    CHAT_REPLY_CACHE[normalized_question] = reply
    return reply


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    return {"reply": get_chat_reply(request.question)}


def validate_upload_file(file: UploadFile) -> None:
    if file.content_type and file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Formato de archivo no válido. Solo se permiten imágenes PNG, JPG o WEBP."
            ),
        )

    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size == 0:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")
    if size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo excede el límite de {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        validate_upload_file(file)
        image = Image.open(file.file)
        return predict_from_image(image)
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Formato de imagen no válido. Usa PNG, JPG o WEBP.",
        )
    except RuntimeError as error:
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


if __name__ == "__main__":
    args = parse_args()

    # Si NO se pasaron argumentos de consola (es decir, ejecucion directa del script como servidor local)
    if args is None:
        import uvicorn
        # Lee el puerto dinámico si existe, o usa el 8000 por defecto
        puerto = int(os.environ.get("PORT", 8000))
        print(f"Iniciando servidor de producción en el puerto {puerto}...")
        uvicorn.run("clasificacion_residuos_ejemplo:app", host="0.0.0.0", port=puerto)
    else:
        if args.command == "train":
            train(args.data, args.epochs)
        elif args.command == "predict":
            predict_file(args.image)
