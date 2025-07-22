import os
import shutil
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, TFBertForSequenceClassification

bert_model = 'bert-large-uncased'
metrics_file = 'classification_metrics.json'
dataset_file_path = './../../../DatasetCreator/Creator/dataset.csv'

def preprocess_sentences_lowercase(df, column='sentence'):
    df[column] = df[column].astype(str).str.lower()
    return df

def encode_examples(texts, labels, tokenizer, max_length=128):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return (
        tf.concat(input_ids, axis=0),
        tf.concat(attention_masks, axis=0),
        tf.convert_to_tensor(labels)
    )

df = pd.read_csv(dataset_file_path)
df = preprocess_sentences_lowercase(df, column='sentence')

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['section'])
label_mapping = dict(enumerate(label_encoder.classes_))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained(bert_model)
train_input_ids, train_attention_masks, train_labels_tensor = encode_examples(train_texts, train_labels, tokenizer)
val_input_ids, val_attention_masks, val_labels_tensor = encode_examples(val_texts, val_labels, tokenizer)

num_classes = len(label_mapping)
model = TFBertForSequenceClassification.from_pretrained(bert_model, num_labels=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    [train_input_ids, train_attention_masks],
    train_labels_tensor,
    validation_data=([val_input_ids, val_attention_masks], val_labels_tensor),
    epochs=1,
    batch_size=32
)

pred_logits = model.predict([val_input_ids, val_attention_masks]).logits
predicted_labels = np.argmax(pred_logits, axis=1)

y_test = np.array(val_labels)
report = classification_report(
    y_test,
    predicted_labels,
    target_names=label_encoder.classes_,
    output_dict=True
)
report['accuracy'] = accuracy_score(y_test, predicted_labels)

with open(metrics_file, 'w') as f:
    json.dump(report, f, indent=4)

model.save_pretrained('Model')
tokenizer.save_pretrained('Model')
