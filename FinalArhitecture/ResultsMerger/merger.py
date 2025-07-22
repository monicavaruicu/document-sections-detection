import os
import pandas as pd

bert_folder_path = './../TrainTestPreprocessingArhitecture/BERTModel/Results'
roberta_folder_path = './../TrainTestPreprocessingArhitecture/RoBERTaModel/Results'
bert_file_name = 'bert_results.csv'
roberta_file_name = 'roberta_results.csv'

fisiere_csv = sorted([f for f in os.listdir(roberta_folder_path) if f.endswith('.csv')])

lista_df = []
for fisier in fisiere_csv:
    cale_completa = os.path.join(roberta_folder_path, fisier)
    df = pd.read_csv(cale_completa)
    lista_df.append(df)

df_final = pd.concat(lista_df, ignore_index=True)

df_final.to_csv(roberta_file_name, index=False)
