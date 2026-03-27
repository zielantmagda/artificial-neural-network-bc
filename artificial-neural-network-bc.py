import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)

FILE_NAME = 'breast_cancer.csv'

try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"ERROR: File '{FILE_NAME}' not found. Make sure it is available.")
    exit()

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


#DATA PREPROCESSING
s_scaler = StandardScaler()
X_train_scaled = s_scaler.fit_transform(X_train)
X_test_scaled = s_scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]
print(f"\nNumber of input features (after processing): {input_dim}")


#DEFINITION AND TRAINING OF THE MODEL (NEURAL NETWORK)
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nStarting model training...")
history = model.fit(X_train_scaled, y_train,
                    epochs=200,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[early_stopping])
print("Training completed.")


#MODEL QUALITY EVALUATION
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.45).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Benign (B)', 'Malignant (M)'])


#RESULTS
print("MODEL QUALITY EVALUATION RESULTS")
print(f"Test set accuracy: {test_accuracy:.4f}")
print("\nClassification Report:\n", class_report)


plt.figure(figsize=(12, 5))

#LOSS PLOT
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(figsize=(12, 5))

#ACCURACY PLOT
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


#HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (B)','Malignant (M)'],
            yticklabels=['Benign (B)','Malignant (M)'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()