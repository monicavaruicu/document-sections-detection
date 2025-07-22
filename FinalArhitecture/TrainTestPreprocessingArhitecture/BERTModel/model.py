import json
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, TFBertForSequenceClassification
from collections import Counter

batch = 1 # change it here

def preprocess_sentences_lowercase(df, column='sentence'):
    df[column] = df[column].astype(str).str.lower()
    return df

def shorten_label(label):
    words = label.strip().split()
    return (words[0][0] + words[1][0]).upper() if len(words) > 1 else words[0][0].upper()

def encode_examples(texts, labels, tokenizer, max_length=128):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_length,
            truncation=True, padding='max_length',
            return_attention_mask=True, return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return (
        tf.concat(input_ids, axis=0),
        tf.concat(attention_masks, axis=0),
        tf.convert_to_tensor(labels)
    )

def encode_texts_only(texts, tokenizer, max_length=128):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_length,
            truncation=True, padding='max_length',
            return_attention_mask=True, return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)

train_df = pd.read_csv('Train/train.csv')
train_df = preprocess_sentences_lowercase(train_df, column='sentence')

test_df = pd.read_csv('Test/test.csv')
test_df = preprocess_sentences_lowercase(test_df, column='sentence')

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['section'])
test_df['label'] = label_encoder.transform(test_df['section'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

X_train, X_val, y_train, y_val = train_test_split(
    train_df['sentence'].tolist(), train_df['label'].tolist(), test_size=0.1, random_state=42
)

train_input_ids, train_attention_masks, train_labels_tensor = encode_examples(X_train, y_train, tokenizer)
val_input_ids, val_attention_masks, val_labels_tensor = encode_examples(X_val, y_val, tokenizer)

num_classes = len(label_encoder.classes_)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

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

model.fit(
    [train_input_ids, train_attention_masks],
    train_labels_tensor,
    validation_data=( [val_input_ids, val_attention_masks], val_labels_tensor ),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

test_df["original_index"] = np.arange(len(test_df))

results = []
all_true = []
all_pred = []

grouped = test_df.groupby('paperName')

for paper_name, group in grouped:
    group = group.reset_index(drop=False)
    num_sentences = len(group)

    for start_idx in range(0, num_sentences, 20):
        end_idx = min(start_idx + 20, num_sentences)
        batch = group.iloc[start_idx:end_idx]

        batch_sentences = batch['sentence'].tolist()
        batch_labels = batch['label'].tolist()
        batch_original_sentences = batch['sentence'].tolist()
        batch_paperNames = [paper_name] * len(batch_sentences)
        batch_indices = batch['original_index'].tolist()

        input_ids, attention_masks = encode_texts_only(batch_sentences, tokenizer)
        preds = model.predict([input_ids, attention_masks]).logits
        pred_classes = np.argmax(preds, axis=1)

        all_true.extend(batch_labels)
        all_pred.extend(pred_classes)

        true_labels_names = label_encoder.inverse_transform(batch_labels)
        pred_labels_names = label_encoder.inverse_transform(pred_classes)

        actual_short = [shorten_label(lbl) for lbl in true_labels_names]
        predicted_short = [shorten_label(lbl) for lbl in pred_labels_names]

        most_common = Counter(predicted_short).most_common(1)[0][0]
        predicted_batch = [most_common] * len(predicted_short)

        for idx, (paper, sent, actual, predicted, predicted_b, original_idx) in enumerate(
            zip(batch_paperNames, batch_original_sentences, actual_short, predicted_short, predicted_batch, batch_indices)
        ):
            results.append({
                "paperName": paper,
                "sentence": sent,
                "actual_class": actual,
                "predicted_class": predicted,
                "predicted_class_batch": predicted_b,
                "original_index": original_idx
            })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="original_index").drop(columns=["original_index"])
results_df.to_csv(f"Results/results_batches_{batch}.csv", index=False, encoding='utf-8')

true_labels = np.array(all_true)
predicted_labels = np.array(all_pred)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=label_encoder.classes_,
    output_dict=True
)
report['accuracy'] = accuracy_score(true_labels, predicted_labels)

with open("bert_classification_metrics.json", "w") as f:
    json.dump(report, f, indent=4)
