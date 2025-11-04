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
async def detectar(image: UploadFile = File(...)):
    try:
        # Leer imagen subida
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Ejecutar predicción
        results = model.predict(img, conf=0.25)

        # Extraer detecciones
        objetos = []
        for box in results[0].boxes:
            clase = model.names[int(box.cls)]
            conf = float(box.conf)
            objetos.append({"clase": clase, "confianza": conf})

        total = len(objetos)
        promedio_confianza = sum(o["confianza"] for o in objetos) / total if total > 0 else 0

        # Retornar resultado en formato JSON
        return JSONResponse(content={
            "detecciones": total,
            "promedio_confianza": promedio_confianza,
            "objetos": objetos
        })
    
    except Exception as e:
        # Mostrar error para depuración
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

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



