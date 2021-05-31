import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import umap

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
print(train_X.shape)

train_X = train_X.astype('float32') / 255.
test_X = test_X.astype('float32') / 255.
train_X = np.reshape(train_X, (len(train_X), 28, 28, 1))
test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))

# input_img = keras.Input(shape=(28, 28, 1))

# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation='relu')(x)
# x = layers.UpSampling2D((2, 2))(x)
# decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoder = keras.Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# print(autoencoder.summary())

# history = autoencoder.fit(train_X, train_X,
#                           epochs=50,
#                           batch_size=128,
#                           shuffle=True,
#                           validation_data=(test_X, test_X))
# autoencoder.save('autoencoder_model.h5')

autoencoder = keras.models.load_model('autoencoder_model.h5')

# input placeholder
inp = autoencoder.input
outputs = [autoencoder.layers[7].output]     					# all layer outputs
functors = [K.function([inp], [out])
            for out in outputs]    # evaluation functions

# get feature representtions
layer_outs = [func([test_X, 1.]) for func in functors]
latent = layer_outs[0][0].reshape(test_X.shape[0], 128)
print(latent.shape)

UMAP = umap.UMAP(n_components=3)
data = UMAP.fit_transform(latent)

out_array = np.empty((data.shape[0], 4))
out_array[:, :3] = data
out_array[:, 3] = test_y
df = pd.DataFrame(out_array, columns=['x', 'y', 'z', 'Value'])
df.to_csv('umap.csv')
