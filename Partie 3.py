# === Étape 1 : Chargement et prétraitement ===
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/Benjamin/tf-exercice/venv/expense-classification.csv").dropna()
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])
num_classes = len(label_encoder.classes_)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Description'], df['label'], test_size=0.2, random_state=42
)

vectorize_layer = layers.TextVectorization(
    max_tokens=10000, output_mode='int', output_sequence_length=50
)
vectorize_layer.adapt(train_texts)

# === Étape 2 : Modèle LSTM ===
model = tf.keras.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    layers.Embedding(input_dim=10001, output_dim=128),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# === Étape 3 : Compilation, callbacks, entraînement ===
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model_nlp.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

model.fit(train_texts, train_labels, validation_data=(test_texts, test_labels), epochs=10, callbacks=callbacks)

loss, acc = model.evaluate(test_texts, test_labels)
print(f"Test Accuracy: {acc:.4f}")
