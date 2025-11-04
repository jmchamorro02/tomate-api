from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO  # si usas YOLOv8, o cambia según tu versión
import cv2
import numpy as np
import io
from PIL import Image
from fastapi.responses import HTMLResponse

app = FastAPI(title="API Detección de Tomates")

# Cargar modelo entrenado (una vez)
model = YOLO("best.pt") # tu modelo YOLOlive o YOLOv8

@app.post("/detectar")
async def detectar_tomates(image: UploadFile = File(...)):
    # Leer imagen
    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Inferencia
    results = model(frame)
    detecciones = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        conf = float(r.conf[0])
        cls = model.names[int(r.cls[0])]
        detecciones.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "class": cls
        })

    # Filtrar sólo tomates (si tu modelo detecta más de una clase)
    tomates = [d for d in detecciones if d["class"].lower() == "tomate"]
    conf_prom = np.mean([d["conf"] for d in tomates]) if tomates else 0


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body>
            <h2>Subir imagen para detección de tomates 🍅</h2>
            <form action="/detectar" enctype="multipart/form-data" method="post">
                <input name="image" type="file" accept="image/*">
                <input type="submit" value="Detectar">
            </form>
        </body>
    </html>
    """
    return JSONResponse({
        "detecciones": len(tomates),
        "promedio_confianza": conf_prom,
        "objetos": tomates
    })


