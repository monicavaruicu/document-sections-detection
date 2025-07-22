import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.combine import SMOTETomek
from tensorflow.keras import Input
import joblib
import json
from sklearn.metrics import accuracy_score

dataset_file_path = './../../../DatasetCreator/Embeddings/embeddings.csv'
metrics_file = 'classification_metrics.json'


df = pd.read_csv(dataset_file_path)

sentences = df['sentence']
sections = df['section']

embeddings = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

label_encoder = LabelEncoder()
sections_encoded = label_encoder.fit_transform(sections)

X = np.vstack(embeddings.values)

X_train, X_test, y_train, y_test = train_test_split(X, sections_encoded, test_size=0.25, random_state=42)
smote_tomek = SMOTETomek(random_state=42)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(len(np.unique(sections_encoded)), activation='softmax'))

optimizer = Adam(learning_rate=1e-4)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop, lr_schedule])

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
    
report = classification_report(
    y_test,
    predicted_labels,
    target_names=label_encoder.classes_,
    output_dict=True,
)

report['accuracy'] = accuracy_score(y_test, predicted_labels)

with open(metrics_file, 'w') as f:
    json.dump(report, f, indent=4)

model.save('Model/dummy_model.keras')