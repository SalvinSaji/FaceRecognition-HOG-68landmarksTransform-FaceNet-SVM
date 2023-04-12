import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

label = []
embedding_vectors = []

def get_image_embeddings(directory_path):
    model = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    image_extensions = ['.jpg', '.jpeg', '.png']
    global label,embedding_vector


    for file in os.listdir(directory_path):
        if os.path.splitext(file)[-1].lower() in image_extensions:
            filename = os.path.join(directory_path, file)
            img = load_img(filename, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            embedding = model.predict(img)[0]
            embedding = embedding.squeeze()[:128]
            embedding = embedding / np.sqrt(np.sum(embedding**2))
            embedding_vectors.append(embedding)
            label.append([directory_path.split('/')[-1]])


def data_embed(dir):
	global label,embedding_vector
	for file in os.listdir(dir):
		get_image_embeddings(f'{dir}/{file}')

data_embed('outdirtrain')
np.save('embeddings.npy',np.array(embedding_vectors))
np.save('labels.npy',np.array(label))
print("Embedding and labes are saved")