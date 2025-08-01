import json
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

roberta_model = 'roberta-base'
metrics_file = 'classification_metrics.json'
dataset_file_path = './../../../DatasetCreator/Creator/dataset.csv'

df = pd.read_csv(dataset_file_path)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['section'])
label_mapping = dict(enumerate(label_encoder.classes_))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42,
)

tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

def encode_examples(texts, labels, max_length=128):
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

train_input_ids, train_attention_masks, train_labels_tensor = encode_examples(train_texts, train_labels)
val_input_ids, val_attention_masks, val_labels_tensor = encode_examples(val_texts, val_labels)

num_classes = len(label_mapping)
model = TFRobertaForSequenceClassification.from_pretrained(roberta_model, num_labels=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    [train_input_ids, train_attention_masks],
    train_labels_tensor,
    validation_data=([val_input_ids, val_attention_masks], val_labels_tensor),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
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