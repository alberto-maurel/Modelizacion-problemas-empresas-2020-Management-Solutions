"""
Concurso de Modelización de Problemas de Empresas UCM 2020
Equipo: YOLO
Integrantes: Alberto Maurel Serrano
             Eduardo Rivero Rodríguez
             Pablo Villalobos Sánchez

Script 2
Precisión: 63,10%
"""

#Importamos las librerías que vamos a utilizar
import math
import numpy as np
import cv2
import keras
from keras import backend as K
import tensorflow as tf
import pickle
import os


#Este código está porque los algoritmos de convolución los ejecutamos sobre una GPU y sin el código nos da problemas
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



#Parámetros del programa
directorio = './Imagenes'
nombre_extractor = '62_26'
n_capa = 13 #Número de capa del modelo de la que se extrae la vectorización
nombre_clasificador = 'RF 62,26 Primera densa'
resolucion_ancho = 960


#Parámetros para dibujar en la imagen
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.7
fontColor              = (0, 0, 255)
lineType               = 2

# Lista que empleamos para escribir las direcciones
emociones = ["enfado", "asco", "miedo", "felicidad", "neutral", "triste", "sorprendido"]







'''
Función que extrae las features de una imagen
Argumentos: X - vector con las imágenes
Return: exTrain - vector con la vectorización de cada imagen
'''
def extract_features(img):
    act = getFeature([img.reshape(1,48,48,1), 0])
    return act





#Cargamos tanto la CNN encargada de extraer las features como el clasificador
#  Esto tarda un poco
extractor = keras.models.load_model(nombre_extractor)
getFeature = K.function([extractor.layers[0].input, K.learning_phase()],
                        [extractor.layers[n_capa].output])

clasificador = pickle.load(open(nombre_clasificador, 'rb'))

# Cargamos el clasificador en cascada que utilizaremos para detectar las caras
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')





for imagen in os.listdir(directorio):
    # Cargamos la imagen
    img = cv2.imread(os.path.join(directorio, imagen))
    
    #La reescalamos a un buen tamaño
    width = resolucion_ancho
    height = math.ceil(img.shape[0] * (resolucion_ancho/img.shape[1]))
    img = cv2.resize(img, (width, height))
    
    # Detectamos las caras
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    
    # Dibujamos alrededor de las caras un rectángulo
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), fontColor, lineType)
        
    
    # Convertimos la imagen a gris (el predictor espera las imágenes en blanco y negro)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    x_pr = np.empty((0,48,48,1))
    
    for (x, y, w, h) in faces:
        # Recortamos la imagen
        img_recortada = img_gray[y:y+h, x:x+w]
        
        # La reescalamos al tamaño que espera la red
        img_recortada = cv2.resize(img_recortada, (48, 48))
        img_recortada = np.reshape(img_recortada, (48,48,1))
    
        #Predecimos el sentimiento de la cara
        #1. Si usamos keras, tenemos que usar model.predict, y tendremos que sacar el argmax
        features = extract_features(img_recortada)
        
        #2. Si usamos una SVM, Random Forest, ... nos saldrá la clase directamente
        sol = clasificador.predict(features[0])
        cv2.putText(img, emociones[int(sol)], (x, y - 5), font, fontScale,fontColor,lineType)
    
    
    
    
    # Imprimimos la foto
    cv2.imshow('img', img)
    cv2.waitKey()