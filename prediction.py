from PIL import Image
from io import BytesIO
import numpy as np  
from numpy.lib.type_check import imag
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions

input_shape = [224, 224]

def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")

    return model

model = load_model()


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image



def predict(image: Image.Image):
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    result = decode_predictions(model.predict(image), 2)[0]
    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)

    return response
