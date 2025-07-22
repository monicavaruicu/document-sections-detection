import pandas as pd
from paperTextContentParser import *
from paperHTMLContentScraper import *

sections_file = 'sections.csv'

def init_sections_csv():
    if os.path.exists(sections_file):
        os.remove(sections_file)

    with open(sections_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['title', 'sections'])

def scrape_papers(inputFile):
    df = pd.read_csv(inputFile)
    init_sections_csv()

    for index, row in df.iterrows():
        urls = row['papers'].split(',')

        for url in urls:
            url = url.strip()
            title = url.split('/')[-1]

            extract_html_content(url, title)
            extract_text_content(url, title, sections_file)
