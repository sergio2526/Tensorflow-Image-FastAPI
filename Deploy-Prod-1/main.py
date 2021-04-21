from fastapi import FastAPI, File, UploadFile
from werkzeug.utils import secure_filename

import json
import numpy as np
from typing import List
from tensorflow.keras.preprocessing import image


# Import model_loader.py functions
from model_loader import loadModelH5


# Main definition for FastAPI
app = FastAPI(
    title="Tensorflow model classification",
    version=1.0,
    description="This is an example",
)


# Variables globales
global loaded_model
loaded_model = loadModelH5()


# Variables constantes
UPLOAD_FOLDER = "uploads/images"
CLASSES = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


# Funcion obtiene el tipo de archivo
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Define a default route
@app.get("/")
def main_page():
    return "REST service is active via FastAPI"


# Metodo que hace las predicciónes via POST
@app.post("/model/predict/")
async def predict(files: List[UploadFile] = File(...)):
    data = {"success": False}

    # Validando y guardando las imagenes recibidas
    data["predictions"] = []
    tmpfiles = []
    for file in files:
        filename = file.filename
        if file and allowed_file(filename):
            print("\nFilename received:", filename)
            contents = await file.read()
            filename = secure_filename(filename)
            tmpfile = "".join([UPLOAD_FOLDER, "/", filename])
            with open(tmpfile, "wb") as f:
                f.write(contents)
            print("Filename stored:", tmpfile)
            tmpfiles.append(tmpfile)

        # Transformando imagen entrante size=224 y escala entre 0 a 1
        image_to_predict = image.load_img(tmpfile, target_size=(224, 224))
        test_image = image.img_to_array(image_to_predict)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.astype("float32")
        test_image /= 255.0

        # Invocando al modelo de keras (H5) para respectiva predicción
        predictions = loaded_model.predict(test_image)[0]
        index = np.argmax(predictions)
        ClassPred = CLASSES[index]
        ClassProb = predictions[index]

        # Resultados Json
        print("Classes:", CLASSES)
        print("Predictions", predictions)
        print("Predicción Index:", index)
        print("Predicción Label:", ClassPred)
        print("Predicción Prob: {:.2%}".format(ClassProb))

        r = {"label": ClassPred, "score": float(ClassProb)}
        data["predictions"].append(r)

        # Success
        data["success"] = True

    return data
