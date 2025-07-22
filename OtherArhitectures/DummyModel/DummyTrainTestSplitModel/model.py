import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Input
from collections import Counter

def shorten_label(label):
    words = label.strip().split()
    if len(words) == 1:
        return words[0][0].upper()
    return (words[0][0] + words[1][0]).upper()

train_df = pd.read_csv('DatasetsCreator/Train/train_embeddings.csv')
test_df = pd.read_csv('DatasetCreatorTest/test_embeddings.csv')

X_train = train_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
X_test = test_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

X_train = np.vstack(X_train.values)
X_test = np.vstack(X_test.values)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['section'])
y_test = label_encoder.transform(test_df['section'])

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

model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(len(np.unique(y_train)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

history = model.fit(X_train_split, y_train_split,
                    validation_data=(X_val_split, y_val_split),
                    epochs=10, 
                    batch_size=32,
                    callbacks=[early_stop, lr_schedule])

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

actual_classes = label_encoder.inverse_transform(y_test)
predicted_classes = label_encoder.inverse_transform(predicted_labels)

actual_short = [shorten_label(lbl) for lbl in actual_classes]
predicted_short = [shorten_label(lbl) for lbl in predicted_classes]

X_test_texts = test_df["sentence"].values

output_df = pd.DataFrame({
    "paperName": test_df["paperName"],
    "sentence": X_test_texts,
    "actual_class": actual_short,
    "predicted_class": predicted_short
})

batch_size = 10
predicted_batch_class = []
i = 0
n = output_df.shape[0]

while i < n:
    batch_end = min(i + batch_size, n)
    current_batch_paper = test_df.iloc[i]['paperName']

    for j in range(i, batch_end):
        if test_df.iloc[j]['paperName'] != current_batch_paper:
            batch_end = j
            break

    batch_predicted_classes = output_df.iloc[i:batch_end]['predicted_class']
    most_common_class = batch_predicted_classes.mode().iloc[0]

    predicted_batch_class.extend([most_common_class] * (batch_end - i))
    i = batch_end

output_df["predicted_class_batch"] = predicted_batch_class
output_df.to_csv("results.csv", index=False, encoding='utf-8')

report = classification_report(
    y_test,
    predicted_labels,
    target_names=label_encoder.classes_,
    output_dict=True,
)

report['accuracy'] = accuracy_score(y_test, predicted_labels)

with open('classification_metrics.json', 'w') as f:
    json.dump(report, f, indent=4)

model.save('Model/dummy_model.keras')
