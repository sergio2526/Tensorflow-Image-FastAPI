from fastapi import FastAPI, File, UploadFile
from werkzeug.utils import secure_filename

import json
import numpy as np
from tensorflow.keras.preprocessing import image


# Import model_loader.py functions
from model_loader import loadModelH5


# Main definition for FastAPI
app = FastAPI(title='Tensorflow model classification',
            version=1.0,
            description="This is an example")


# Variables globales
global loaded_model
loaded_model = loadModelH5()


# Folde para guardar las imagenes cargadas
UPLOAD_FOLDER = "uploads/images"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


# Funcion obtiene el tipo de archivo
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Define a default route
@app.get("/")
def main_page():
    return "REST service is active via FastAPI"


@app.post("/model/predict/")
async def predict(file: UploadFile = File(...)):
    data = {"success": False}
    filename = file.filename
    if file and allowed_file(filename):
        print("\nFilename received:", filename)
        contents = await file.read()
        filename = secure_filename(filename)
        tmpfile = "".join([UPLOAD_FOLDER, "/", filename])
        with open(tmpfile, "wb") as f:
            f.write(contents)
        print("\nFilename stored:", tmpfile)

        # Transformando imagen entrante size=224 y escala entre 0 a 1
        image_to_predict = image.load_img(tmpfile, target_size=(224, 224))
        test_image = image.img_to_array(image_to_predict)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.astype("float32")
        test_image /= 255.0

        predictions = loaded_model.predict(test_image)[0]
        index = np.argmax(predictions)
        CLASSES = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
        ClassPred = CLASSES[index]
        ClassProb = predictions[index]

        # Results as Json
        for i in range(2):
            if ClassProb >= 0.20:

                print("Classes:", CLASSES)
                print("Predictions", predictions)
                print("Predicción Index:", index)
                print("Predicción Label:", ClassPred)
                print("Predicción Prob: {:.2%}".format(ClassProb))
                data["predictions"] = []
                r = {"label": ClassPred, "score": float(ClassProb)}
                data["predictions"].append(r)
            else:
                pass

        # Success
        data["success"] = True

    return data
