import csv

from sentenceSplitter import split_into_sentences

with open('sections_content.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    with open('train.csv', 'w', newline='', encoding='utf-8') as output_file:
        fieldnames = ['paperName', 'sentence', 'section']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        writer.writeheader()
        
        for row in reader:
            paper_name = row['paperName']
            section_content = row['sectionContent']
            section_name = row['sectionName']
            
            sentences = split_into_sentences(section_content)
            
            for sentence in sentences:
                if len(sentence.strip()) > 5:
                    writer.writerow({
                        'paperName': paper_name,
                        'sentence': sentence,
                        'section': section_name
                    })
