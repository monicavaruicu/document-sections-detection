import pandas as pd
import json

base_name = 'results_batches'
file_number = '_1'
file_name = base_name + file_number

df = pd.read_csv(f'./../{file_name}.csv')

texts = df['sentence'].astype(str)
lungimi_caractere = texts.str.len()
lungimi_cuvinte = texts.str.split().apply(len)

media_lungime_cuvinte = lungimi_cuvinte.mean()
media_lungime_caractere = lungimi_caractere.mean()

analiza_lungimi = {
    'total_instante': len(df),
    'total_cuvinte': int(lungimi_cuvinte.sum()),
    'total_caractere': int(lungimi_caractere.sum()),
    'media_lungime_fraza_in_cuvinte': round(media_lungime_cuvinte, 2),
    'media_lungime_fraza_in_caractere': round(media_lungime_caractere, 2),
    'lungime_maxima_fraza_caractere': int(lungimi_caractere.max()),
    'lungime_minima_fraza_caractere': int(lungimi_caractere.min()),
    'lungime_maxima_fraza_cuvinte': int(lungimi_cuvinte.max()),
    'lungime_minima_fraza_cuvinte': int(lungimi_cuvinte.min()),
    'fraze_peste_media_cuvinte': int((lungimi_cuvinte > media_lungime_cuvinte).sum()),
    'fraze_sub_media_cuvinte': int((lungimi_cuvinte < media_lungime_cuvinte).sum()),
    'fraze_peste_media_caractere': int((lungimi_caractere > media_lungime_caractere).sum()),
    'fraze_sub_media_caractere': int((lungimi_caractere < media_lungime_caractere).sum())
}

distributii_etichete = {}
for coloana in ['actual_class', 'predicted_class', 'predicted_class_batch']:
    distributii_etichete[coloana] = df[coloana].value_counts().to_dict()

rezultat_final = {
    'analiza_lungimi_fraze': analiza_lungimi,
    'distributii_pe_clase': distributii_etichete
}

with open(f'{file_name}.json', 'w', encoding='utf-8') as f:
    json.dump(rezultat_final, f, indent=4, ensure_ascii=False)

