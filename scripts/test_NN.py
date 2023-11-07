import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity


N = 20
M = 10
num_samples = 1000
#input_data = np.random.rand(num_samples, N)
input_data = [np.random.rand(5, 5) for _ in range(num_samples)]
input_data = np.asarray(input_data)

print(input_data.shape)

exit()
input_layer = keras.layers.Input(shape=(N,))
encoder = keras.layers.Dense(64, activation='relu')(input_layer)
encoder = keras.layers.Dense(32, activation='relu')(encoder)
encoded = keras.layers.Dense(M, activation='relu')(encoder)

decoder = keras.layers.Dense(32, activation='relu')(encoded)
decoder = keras.layers.Dense(64, activation='relu')(decoder)
decoded = keras.layers.Dense(N)(decoder)

autoencoder = keras.models.Model(inputs=input_layer,
                                 outputs=decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(input_data, input_data, epochs=10)

input_data2 = np.random.rand(num_samples, N)

autoencoder.fit(input_data2, input_data2, epochs=10)

encoded_vectors = keras.Model(inputs=input_layer, outputs=encoded).predict(input_data)
