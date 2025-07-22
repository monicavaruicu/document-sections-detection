import pandas as pd
from mistralai import Mistral
import time
import os

os.environ["MISTRAL_API_KEY"] = "RscFGcvR47zUkZ1rtm3pjTriz5G2w1Cz"

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"
input_file = 'train.csv'
output_file = 'train_embeddings.csv'

df = pd.read_csv(input_file)

sentences = df['sentence']
sections = df['section']

batch_size = 200
pause_duration = 5
embeddings = []

client = Mistral(api_key=api_key)

for idx in range(0, len(sentences), batch_size):
    batch_sentences = sentences[idx:idx+batch_size].tolist()
    response = client.embeddings.create(
        model=model,
        inputs=batch_sentences
    )
    batch_embeddings = [embedding.embedding for embedding in response.data]
    embeddings.extend(batch_embeddings)
    print(f"Batch {idx//batch_size + 1} processed.")
    time.sleep(pause_duration)

embeddings_df = pd.DataFrame({
    'sentence': sentences,
    'section': sections,
    'embedding': embeddings
})

embeddings_df.to_csv(output_file, index=False)
