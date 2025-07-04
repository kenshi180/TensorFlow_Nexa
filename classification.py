import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
from sklearn.metrics import accuracy_score

# Chargement et préparation
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = resize(X_train, (224, 224)).numpy()
X_test = resize(X_test, (224, 224)).numpy()

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Création modèle simple
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Entraînement rapide pour capture écran...")
model.fit(X_train, y_train_cat, epochs=3, batch_size=64, validation_split=0.2, verbose=0)

# Évaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = y_test.flatten()
acc = accuracy_score(y_true, y_pred)
loss = model.evaluate(X_test, y_test_cat, verbose=0)[0]

# Résultat simulé pour capture
df = pd.DataFrame([{
    "optimizer": "adam",
    "learning_rate": 0.001,
    "epochs": 3,
    "test_accuracy": acc,
    "test_loss": loss
}])

print("\n📸 Tableau comparatif (version rapide) :")
print(df)

# Graphique pour capture écran n°9
plt.figure(figsize=(6, 3))
plt.title("Comparaison des modèles")
plt.bar(["adam, 0.001"], df["test_loss"])
plt.ylabel("Test Loss")
plt.tight_layout()
plt.show()
