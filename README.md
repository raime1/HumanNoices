# HumanNoices

Proyecto para reconocer si un sonido es de procedencia humana o no. Basado en la [CNN de Reconocimeinto de Generos Musicales](https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/16-%20How%20to%20implement%20a%20CNN%20for%20music%20genre%20classification/code/cnn_genre_classifier.py) de Valerio Velardo y utiliza el dataset [ESC-50](https://github.com/karolpiczak/ESC-50)

### Sonidos Reconocibles

* Bebe llorando
* Estornudo
* Aplausos
* Respirando
* Tos
* Pasos
* Risas
* Lavandose los dientes
* Ronquidos
* Beber algo

## Como usar

> python recognize_sound.py -f audiofile.wav

Se incluyeron varios archvios de audios en la carpeta examples para poder probar el modelo de manera rápida.

## Rendimiento

Se redujó el número de capas de convoluciones de 3 a 1. El modelo mostró mejor rendimiento con una sola capa.

#### 1 Capa

![1 Capa](https://github.com/raime1/HumanNoices/blob/main/model_test_1conv.png) 

#### 2 Capas

![2 Capas](https://github.com/raime1/HumanNoices/blob/main/model_test_2conv.png) 

#### 3 Capas

![3 Capas](https://github.com/raime1/HumanNoices/blob/main/model_test_3conv.png) 

## Sources

* https://www.researchgate.net/publication/330247633_Sound_Classification_Using_Convolutional_Neural_Network_and_Tensor_Deep_Stacking_Network
* https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/12-%20Music%20genre%20classification:%20Preparing%20the%20dataset/code/extract_data.py#L45
* https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/16-%20How%20to%20implement%20a%20CNN%20for%20music%20genre%20classification/code/cnn_genre_classifier.py
* https://github.com/karolpiczak/ESC-50
