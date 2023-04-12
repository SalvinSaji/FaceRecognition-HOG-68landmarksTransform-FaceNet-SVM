import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

embedding_vectors = []

def get_image_embeddings(directory_path):
    model = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    image_extensions = ['.jpg', '.jpeg', '.png']
    global embedding_vector
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

def data_embed(dir):
	global embedding_vector
	for file in os.listdir(dir):
		get_image_embeddings(f'{dir}/{file}')


data_embed('outdirtest')
np.save('test_embeddings.npy',np.array(embedding_vectors))
print("Embedding saved successfully")
