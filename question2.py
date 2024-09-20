from google.colab import drive
drive.mount('/content/drive')
q1_dir = '/content/drive/My Drive/KITTI-Sequence'
q2_dir = '/content/drive/My Drive/img_dtlabs'

from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Lambda, Input
from tensorflow.keras import backend as K
import numpy as np
import cv2
import os
import pandas as pd

def load_images_from_directory(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((file, img))
                else:
                    print(f"Erro ao carregar a imagem: {img_path}")
    return images

images = load_images_from_directory(extract_dir)

if len(images) > 0:
    print(f"Total de imagens encontradas: {len(images)}")
else:
    print("Nenhuma imagem encontrada no diretório ou subdiretórios.")

descriptor_size = 128
shape = 50
activation = 'relu6'

# Definindo o modelo
model = Sequential()
model.add(Input(shape=(shape, shape, 3)))
model.add(Conv2D(16, kernel_size=(3, 3), activation=activation))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(descriptor_size))
model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
model.compile(optimizer='Adam', loss='mse')

def get_descriptor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (shape, shape))
    img = np.expand_dims(img, axis=0) / 255.0
    descriptor = model.predict(img)
    return descriptor

database = {}

def add_to_database(name, image_path):
    descriptor = get_descriptor(image_path)
    database[name] = descriptor

def recognize_face(test_image_path):
    test_descriptor = get_descriptor(test_image_path)
    closest_name = None
    closest_distance = float('inf')

    for name, descriptor in database.items():
        distance = np.linalg.norm(test_descriptor - descriptor)
        if distance < closest_distance:
            closest_distance = distance
            closest_name = name

    return closest_name, closest_distance

add_to_database('celebrity_1', 'path/to/celebrity_1_image.jpg')
add_to_database('celebrity_2', 'path/to/celebrity_2_image.jpg')

recognized_name, score = recognize_face('path/to/image_with_mask.jpg')
print(f'Pessoa reconhecida: {recognized_name}, Score: {score}')

infer_image = cv2.imread('path/to/image_with_mask.jpg')
cv2.imshow('Imagem para Inferência', infer_image)
cv2.waitKey(0)
cv2.destroyAllWindows()