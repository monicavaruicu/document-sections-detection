import csv

from Splitters.sectionSplitter import split_into_sections
from Splitters.sentenceSplitter import split_into_sentences

sections_content_file = 'sections_content.csv'
dataset_file = 'test.csv'

def main():
    split_into_sections(sections_content_file)

    with open(sections_content_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        with open(dataset_file, 'w', newline='', encoding='utf-8') as output_file:
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
                        
if __name__ == '__main__':
    main()

