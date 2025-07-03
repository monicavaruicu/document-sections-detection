import pandas as pd
import json

df = pd.read_csv('./../dataset_clean_roberta.csv')

total_instante = len(df)

distributie_section = {str(k): int(v) for k, v in df['section'].value_counts().to_dict().items()}

texts = df['sentence'].astype(str)

lungimi_caractere = texts.str.len()
lungimi_cuvinte = texts.str.split().apply(len)

total_caractere = int(lungimi_caractere.sum())
total_cuvinte = int(lungimi_cuvinte.sum())

media_lungime_cuvinte = lungimi_cuvinte.mean()
media_lungime_caractere = lungimi_caractere.mean()

lungime_max_caractere = int(lungimi_caractere.max())
lungime_min_caractere = int(lungimi_caractere.min())

lungime_max_cuvinte = int(lungimi_cuvinte.max())
lungime_min_cuvinte = int(lungimi_cuvinte.min())

numar_fraze_peste_media_cuvinte = int((lungimi_cuvinte > media_lungime_cuvinte).sum())
numar_fraze_sub_media_cuvinte = int((lungimi_cuvinte < media_lungime_cuvinte).sum())

numar_fraze_peste_media_caractere = int((lungimi_caractere > media_lungime_caractere).sum())
numar_fraze_sub_media_caractere = int((lungimi_caractere < media_lungime_caractere).sum())

rezultat = {
    'total_instante': total_instante,
    'total_cuvinte': total_cuvinte,
    'total_caractere': total_caractere,
    'media_lungime_fraza_in_cuvinte': round(media_lungime_cuvinte, 2),
    'media_lungime_fraza_in_caractere': round(media_lungime_caractere, 2),
    'lungime_maxima_fraza_caractere': lungime_max_caractere,
    'lungime_minima_fraza_caractere': lungime_min_caractere,
    'lungime_maxima_fraza_cuvinte': lungime_max_cuvinte,
    'lungime_minima_fraza_cuvinte': lungime_min_cuvinte,
    'fraze_peste_media_cuvinte': numar_fraze_peste_media_cuvinte,
    'fraze_sub_media_cuvinte': numar_fraze_sub_media_cuvinte,
    'fraze_peste_media_caractere': numar_fraze_peste_media_caractere,
    'fraze_sub_media_caractere': numar_fraze_sub_media_caractere,
    'distributie_pe_sectiuni': distributie_section
}

with open('dataAnalysis.json', 'w', encoding='utf-8') as f:
    json.dump(rezultat, f, indent=4, ensure_ascii=False)
