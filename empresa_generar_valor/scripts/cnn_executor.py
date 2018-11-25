# Credits:
# https://link.springer.com/article/10.1007/s10278-018-0079-6
# https://github.com/ImagingInformatics/machine-learning
# https://github.com/paras42/Hello_World_Deep_Learning

# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.optimizers import Adam

def cargarModelo():
    json_file = open('../model/nn_model_7v.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../model/nn_model_7v.h5")
    print("Cargando modelo desde el disco ...")
    loaded_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo cargado de disco!")
    graph = tf.get_default_graph()
    return loaded_model, graph
	
def cargarModelo_4v():
    json_file = open('../model/nn_modelo_4v.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../model/nn_modelo_4v.h5")
    print("Cargando modelo desde el disco ...")
    loaded_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo cargado de disco!")
    graph = tf.get_default_graph()
    return loaded_model, graph	

def cargarModelo_3v():
    json_file = open('../model/nn_modelo_3v.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../model/nn_modelo_3v.h5")
    print("Cargando modelo desde el disco ...")
    loaded_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo cargado de disco!")
    graph = tf.get_default_graph()
    return loaded_model, graph