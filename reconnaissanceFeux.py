import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Créer un générateur de données à partir des fichiers d'images
data_generator = tf.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Charger les données d'entraînement et de validation à partir des fichiers
train_generator = data_generator.flow_from_directory(
    ['data/train',]  # Répertoire de fichiers d'images d'entraînement
    target_size=(32, 32),  # Redimensionner les images à 32x32
    batch_size=32,  # Fractionner les données en lots de 32 exemples
    class_mode='binary',  # Utiliser des étiquettes binaires (feu rouge ou vert)
    subset='training')  # Utiliser les données d'entraînement

validation_generator = data_generator.flow_from_directory(
    'data/train',  # Répertoire de fichiers d'images de validation
    target_size=(32, 32),  # Redimensionner les images à 32x32
    batch_size=32,  # Fractionner les données en lots de 32 exemples
    class_mode='binary',  # Utiliser des étiquettes binaires (feu rouge ou vert)
    subset='validation')  # Utiliser les données de validation

# Charger les données de test à partir des fichiers
test_generator = data_generator.flow_from_directory(
    'data/test',  # Répertoire de fichiers d'images de test
    target_size=(32, 32),  # Redimensionner les images à 32x32
    batch_size=32,  # Fractionner les données en lots de 32 exemples
    class_mode='binary')  # Utiliser des étiquettes binaires (feu rouge ou vert)

# Afficher le nombre d'exemples d'entraînement, de validation et de test
print(train_generator.n)
print(validation_generator.n)
print(test_generator.n)

# Chargement des données d'entraînement et de test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalisation des données d'entraînement et de test
x_train = x_train / 255.0
x_test = x_test / 255.0

# Création du modèle CNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Évaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Prédiction sur une image de test
img = x_test[0]
prediction = model.predict(np.expand_dims(img, axis=0))
predicted_class = np.argmax(prediction)
print('Predicted class:', predicted_class)

# Affichage de l'image et de la prédiction
plt.imshow(img)
plt.title('Prediction: ' + str(predicted_class))
plt.show()
