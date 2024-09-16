from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = YOLO('./best v3(100).pt')

@app.get("/")
def read_root():
    return {"message": "Welcome to Water Meter Detection API"}

@app.post("/detect/")
async def detect_water_meter(
    file: UploadFile = File(...)
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    img_array = np.array(image)

    results = model.predict(source=img_array, conf=0.25, imgsz=640)

    detected_numbers = []
    for result in results:
        for box in result.boxes:
            label = result.names[box.cls[0].item()]
            x1, _, _, _ = box.xyxy[0].tolist()

            if label.isdigit():
                detected_numbers.append((x1, label))

    detected_numbers.sort(key=lambda x: x[0])
    sorted_numbers = [num for _, num in detected_numbers]
    detected_values_str = ''.join(sorted_numbers)

    return JSONResponse(
        content={
            "status": "success",
            "detected_values": detected_values_str if sorted_numbers else "No numeric values detected."
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8800)
