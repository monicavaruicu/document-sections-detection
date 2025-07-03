import pandas as pd
import numpy as np
import tensorflow as tf
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def preprocess_sentences_lowercase(df, column='sentence'):
    df[column] = df[column].astype(str).str.lower()
    return df

def encode_examples(texts, tokenizer, max_length=128):
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
    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)

def predict_labels(test_file, model_path, labels_file, output_file, metrics_file="metrics.json"):
    test_df = pd.read_csv(test_file)
    test_df = preprocess_sentences_lowercase(test_df, column='sentence')

    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    with open(labels_file, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(label_mapping.values()))

    input_ids, attention_masks = encode_examples(test_df['sentence'].tolist(), tokenizer)
    pred_logits = model.predict([input_ids, attention_masks]).logits
    predicted_ids = np.argmax(pred_logits, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_ids)

    test_df['predicted_label'] = predicted_labels

    if 'section' in test_df.columns:
      y_true = test_df['section'].values
      report = classification_report(
          y_true,
          predicted_labels,
          target_names=label_encoder.classes_,
          output_dict=True
      )
      report['accuracy'] = accuracy_score(y_true, predicted_labels)
      with open(metrics_file, "w", encoding="utf-8") as f:
          json.dump(report, f, indent=4, ensure_ascii=False)

    test_df.to_csv(output_file, index=False)
