import tensorflow as tf


# Función que carga el modelo H5
def loadModelH5():
    MODEL_H5_FILE = "/home/sergio/Descargas/flowers_model_full_tf2.h5"
    # Cargar el modelo DL desde disco
    loaded_model = tf.keras.models.load_model(MODEL_H5_FILE)
    print(MODEL_H5_FILE, " Loading from disk >> ", loaded_model)

    return loaded_model
