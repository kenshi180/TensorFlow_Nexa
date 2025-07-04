import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Chargement du dataset
df = pd.read_csv("C:/Users/Benjamin/tf-exercice/venv/creditcard.csv")

# 2.3.1 : Chargement et Analyse Exploratoire (EDA)
print("Extrait des données (head) :")
print(df.head(), "\n")

print("Informations sur les données (info) :")
print(df.info(), "\n")

print("Statistiques descriptives (describe) :")
print(df.describe(), "\n")

print("Répartition des classes :")
print(df['Class'].value_counts(), "\n")

sns.countplot(x='Class', data=df)
plt.title('Répartition des classes (0 = Non-fraude, 1 = Fraude)')
plt.show()

# 2.3.2 : Préparation des données
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(columns=['Amount'])

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Taille du train set : {X_train.shape[0]} échantillons")
print(f"Taille du test set : {X_test.shape[0]} échantillons")
print(f"\nRépartition des classes dans le train set :\n{y_train.value_counts(normalize=True)}")
print(f"\nRépartition des classes dans le test set :\n{y_test.value_counts(normalize=True)}")

# 2.3.3 : Entraînement du modèle
clf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

clf.fit(X_train, y_train)
print("\nModèle RandomForest entraîné avec class_weight='balanced'.")

# Évaluation : prédiction sur test
y_pred = clf.predict(X_test)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Non-fraude', 'Fraude']))

# 2.3.4 : Évaluation rigoureuse avec heatmap

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Non-fraude', 'Fraude'],
    yticklabels=['Non-fraude', 'Fraude']
)
plt.title('Matrice de confusion (heatmap)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# Rapport de classification détaillé
print("Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Non-fraude', 'Fraude']))
