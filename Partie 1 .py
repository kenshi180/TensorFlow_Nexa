import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                                     GlobalAveragePooling2D, Dropout)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, Adamax
from tensorflow.keras.backend import clear_session
from tensorflow.image import resize
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import datetime
import os

# 1.2 : Chargement du Jeu de Données CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 1.3.1 : Exploration Visuelle
plt.imshow(X_train[5])
plt.title(f"Classe : {y_train[5][0]}")
plt.axis('off')
plt.show()

# 1.3.2 : Prétraitement des Données
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print("Prétraitement terminé.")
print("X_train:", X_train_norm.shape, "y_train:", y_train_cat.shape)

# === TensorBoard Callback pour le CNN ===
log_dir_cnn = "logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_cnn = TensorBoard(log_dir=log_dir_cnn, histogram_freq=1)

# 1.3.3 : Modèle CNN Basique
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 1.3.4 : Compilation et Entraînement
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_norm, y_train_cat,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[tensorboard_callback_cnn])

# 1.3.5 : Visualisation des Performances
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 1.3.6 : Matrice de Confusion du CNN
y_pred = model.predict(X_test_norm)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred_classes)
print("\nMatrice de confusion - CNN :")
print(cm)

# 1.3.7 : Apprentissage par Transfert avec MobileNetV2

X_train_resized = resize(X_train_norm, (224, 224)).numpy()
X_test_resized = resize(X_test_norm, (224, 224)).numpy()

X_train_preprocessed = preprocess_input(X_train_resized)
X_test_preprocessed = preprocess_input(X_test_resized)

base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

transfer_model = Model(inputs=base_model.input, outputs=output)

log_dir_transfer = "logs/transfer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_transfer = TensorBoard(log_dir=log_dir_transfer, histogram_freq=1)

transfer_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

history_transfer = transfer_model.fit(X_train_preprocessed, y_train_cat,
                                      epochs=5,
                                      batch_size=64,
                                      validation_split=0.2,
                                      callbacks=[tensorboard_callback_transfer])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_transfer.history['accuracy'], label='Accuracy')
plt.plot(history_transfer.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transfer.history['loss'], label='Loss')
plt.plot(history_transfer.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred_transfer = transfer_model.predict(X_test_preprocessed)
y_pred_classes_transfer = np.argmax(y_pred_transfer, axis=1)

cm_transfer = confusion_matrix(y_true, y_pred_classes_transfer)
print("\nMatrice de confusion - Transfer Learning :")
print(cm_transfer)

#1.3.8 Tensorboard

log_dir_transfer = "logs/transfer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_transfer = TensorBoard(log_dir=log_dir_transfer, histogram_freq=1)

transfer_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

history_transfer = transfer_model.fit(X_train_preprocessed, y_train_cat,
                                      epochs=5,
                                      batch_size=64,
                                      validation_split=0.2,
                                      callbacks=[tensorboard_callback_transfer])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_transfer.history['accuracy'], label='Accuracy')
plt.plot(history_transfer.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transfer.history['loss'], label='Loss')
plt.plot(history_transfer.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred_transfer = transfer_model.predict(X_test_preprocessed)
y_pred_classes_transfer = np.argmax(y_pred_transfer, axis=1)

cm_transfer = confusion_matrix(y_true, y_pred_classes_transfer)
print("\nMatrice de confusion - Transfer Learning :")
print(cm_transfer)

#1.3.9 – Grid Search simplifié sur les hyperparamètres

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

def build_cnn_model():
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

optimizers = {
    'adam': Adam,
    'sgd': lambda lr: SGD(learning_rate=lr, momentum=0.9),
    'adamax': Adamax
}
epochs_list = [20, 30, 40, 50, 60]
learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05]

results = []

for opt_name, opt_func in optimizers.items():
    for lr in learning_rates:
        for epochs in epochs_list:
            clear_session()

            print(f"\n Training with: optimizer={opt_name}, lr={lr}, epochs={epochs}")
            model = build_cnn_model()
            optimizer = opt_func(lr)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            log_dir = f"logs/{opt_name}_lr{lr}_ep{epochs}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(f"best_{opt_name}_lr{lr}_ep{epochs}.keras", save_best_only=True),
                TensorBoard(log_dir=log_dir)
            ]

            history = model.fit(X_train, y_train_cat,
                                validation_split=0.2,
                                epochs=epochs,
                                batch_size=64,
                                verbose=0,
                                callbacks=callbacks)

            test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
            results.append({
                'optimizer': opt_name,
                'learning_rate': lr,
                'epochs': epochs,
                'val_acc': max(history.history['val_accuracy']),
                'val_loss': min(history.history['val_loss']),
                'test_acc': test_acc,
                'test_loss': test_loss
            })

df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by='test_loss').reset_index(drop=True)

print("\n Top 10 configurations (triées par test_loss):")
print(df_results_sorted.head(10))

top50 = df_results_sorted.head(50)
best_opt = top50['optimizer'].value_counts().idxmax()
best_lr = top50['learning_rate'].value_counts().idxmax()
best_epochs = top50['epochs'].value_counts().idxmax()
adam_count = top50[top50['optimizer'] == 'adam'].shape[0]

print(f"\n Analyse des 50 meilleures combinaisons :")
print(f"- Optimizer le plus fréquent : {best_opt}")
print(f"- Learning rate le plus fréquent : {best_lr}")
print(f"- Epochs les plus fréquents : {best_epochs}")
print(f"- Nombre de fois où 'adam' est utilisé : {adam_count}")

df_results_sorted.to_csv("resultats_hyperparametres.csv", index=False)