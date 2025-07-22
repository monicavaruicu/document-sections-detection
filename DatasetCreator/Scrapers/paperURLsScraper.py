from urllib.parse import urlencode
import pandas as pd
import requests
from bs4 import BeautifulSoup

parser = 'html.parser'

def extract_papers_url(url, category, data):
    
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, parser)
        dl_tag = soup.find('dl', {'id': 'articles'})
        
        if dl_tag:
            dt_tags = dl_tag.find_all('dt')
            
            if dt_tags:
                paper_urls = []
                for dt in dt_tags:
                    a_tags = dt.find_all('a')
                    
                    for a in a_tags:
                        if a.get('href') and 'html' in a['href']:
                            paper_urls.append(a['href'])
                
                data.append({'category': category, 'papers': ', '.join(paper_urls)})
    else:
        print(f"Error {response.status_code} for {category} URL {url}")

def scrape_papers_urls(inputFile, outputFile):
    df = pd.read_csv(inputFile) 
    data = []

    for index, row in df.iterrows():
        extract_papers_url(row['url'], row['category'], data)

    result_df = pd.DataFrame(data)
    result_df.to_csv(outputFile, index=False)
