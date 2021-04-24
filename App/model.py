import tensorflow as tf

# Función que carga el modelo H5
def loadModelH5():
    MODEL_H5_FILE = "/home/sergio/Descargas/flowers.h5"
    # Cargar el modelo
    loaded_model = tf.keras.models.load_model(MODEL_H5_FILE)
    return loaded_model
