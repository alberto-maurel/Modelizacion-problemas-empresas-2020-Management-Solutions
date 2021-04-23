# Modelizacion-problemas-empresas-2020-Management-Solutions

<b>English version below</b>

<b>Español</b> <br>
Este repositorio contiene la solución presentada al concurso de Modelización de Empresas UCM 2020 - Management Solutions. El reto consistía en detectar las emociones faciales en una serie de imágenes de caras (correspondientes al dataset de la competición FER 2012). 

Nuestra solución emplea VGGFace 2 para extraer un embedding de las imágenes, y posteriormente utilizamos un clasificador custom sobre los embeddings para predecir a qué clase de las 7 pertenece la imagen. Se prueba con varios clasificadores: Random Forest, K-nn neighbours y SVM y varios parámetros para cada uno de ellos, y nos quedamos con la mejor combinación. Con esta solución logramos un <b>68,93% de precisión (el state of the art actualemente logra un 75,42% en dicho dataset)</b>. 

También proponemos otra solución más sencilla con una red neuronal custom sustituyendo a VGGFace2, que logra un 63,10% de precisión y tan solo necesita 1 minuto de entrenamiento (frente a las 3h 25min de la otra solución). Toda la información y desarrollo del proyecto lo tenéis en <i>memoria.pdf</i>, donde explicamos toda la lógica detrás de la solución y cómo funciona.

Para ejecutar el código, está toda la información en <i>leeme.pdf</i>, incluyendo la jerarquía de archivos esperados. Varios de los ficheros necesarios para la ejecución no están disponibles. Más concretamente, faltan los siguientes elementos:
<ul>
  <li> Dataset: es el dataset de la competición FER 2012 de reconocimiento de expresiones faciales. Se puede encontrar fácilmente en internet.
  <li> Clasificador en cascada para extraer las caras: es el archivo <i>haarcascade_frontalface_default.xml</i>, que se puede encontrar en la página de cv2 de CascadeClassifiers.
  <li> Modelos con los pesos: a pesar de tener los ficheros con los pesos de la red neuronal guardados, son archivos grandes que no pueden estar en Github. Por ello tendréis que entrenar vosotros la red y cargar vuestros pesos (no lleva más de 2 horas en un ordenador doméstico).
</ul>

Por último, en el desafío se pedía que se pudiese aplicar sobre fotos genéricas, y que se detectasen primero las caras para después aplicar el algoritmo. Para recortar las caras se emplea un clasificador en cascada de la biblioteca OpenCV.

<hr>

<b>English</b> <br>
This repository contains our solution to the Business Modelling UCM Contest 2020 - Management Solutions. All the materials are written in Spanish. The challenge was to detect the facial expressions from face images (corresponding to the FER 2012 expression recognition dataset).

Our solution consisted of two steps. First, we obtained an embedding of our images with the VGGFace 2 neural network. Then, we tried a bunch of custom classifiers to assign an expression to the embedded images. We tried several classifiers: Random Forest, K-nn neighbours and SVM, and fine-tuned each ones parameters. Then, we select the best combination. <b>This achieves a 68.93% of accuracy (the state of the art achieves a 75.42% in the same dataset)</b>.

We also propose a simpler solution, with a handcrafted neural network to substitute VGGFace 2, achieving a 63.10% of accuracy, with just 1 minute of training (the other architecture requires 3h 25mins). All the information and detailed explanations of the project can be found at <i>memory.pdf</i>.

In order to execute the code, the steps needed are explained in <i>leeme.pdf</i>, including the expected folder hierarchy. Some of the files/data are not available at the repository. More precisely, the non-available materials are:

<ul>
  <li> Dataset: FER 2012 facial expression recognition dataset. It can be easily found on the Internet.
  <li> CascadeClassifier to crop the faces from a photo: it's the file called <i>haarcascade_frontalface_default.xml</i>. It can be found on the OpenCV webpage.
  <li> Weights of the neural networks and pickled classifiers: they are huge files that can't be uploaded to GitHub. However, the training files are provided, and you can execute them on your computer to train/fit them.
</ul>

Lastly, the challenge also asked to apply the model to random photos. So, a first step to crop the faces was needed (that's why we needed the CascadeClassifier).
