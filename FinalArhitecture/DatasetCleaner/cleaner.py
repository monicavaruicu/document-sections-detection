import csv

bert_path = './../ResultsMerger/bert_results.csv'
roberta_path = './../ResultsMerger/roberta_results.csv'
output_bert_path = 'dataset_clean_bert.csv'
output_roberta_path = 'dataset_clean_roberta.csv'

with open(bert_path, newline='', encoding='utf-8') as bert_file, \
     open(roberta_path, newline='', encoding='utf-8') as roberta_file, \
     open(output_bert_path, 'w', newline='', encoding='utf-8') as out_bert, \
     open(output_roberta_path, 'w', newline='', encoding='utf-8') as out_roberta:

    bert_reader = csv.DictReader(bert_file)
    roberta_reader = csv.DictReader(roberta_file)

    fieldnames = ['paperName', 'sentence', 'section']
    bert_writer = csv.DictWriter(out_bert, fieldnames=fieldnames)
    roberta_writer = csv.DictWriter(out_roberta, fieldnames=fieldnames)

    bert_writer.writeheader()
    roberta_writer.writeheader()

    for bert_row, roberta_row in zip(bert_reader, roberta_reader):
        if (bert_row['paperName'].strip().lower() == roberta_row['paperName'].strip().lower() and
            bert_row['sentence'].strip().lower() == roberta_row['sentence'].strip().lower()):
            
            if bert_row['predicted_class'].strip() == roberta_row['predicted_class'].strip():
                bert_writer.writerow({
                    'paperName': bert_row['paperName'],
                    'sentence': bert_row['sentence'],
                    'section': bert_row['predicted_class']
                })
                roberta_writer.writerow({
                    'paperName': roberta_row['paperName'],
                    'sentence': roberta_row['sentence'],
                    'section': roberta_row['predicted_class']
                })
        else:
            print("The input files don't match!!!")
