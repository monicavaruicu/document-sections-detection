from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup
import csv

baseUrl = 'https://arxiv.org/'
parser = 'html.parser'

def scrape_categories(subject, outputFile):

    with open(outputFile, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['category', 'url', 'noOfPapers'])

        response = requests.get(baseUrl)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, parser)

            section = soup.find('h2', string=subject)
            if section:
                ul = section.find_next('ul')
                links = ul.find_all('a')

                for link in links[5:]:
                    sub_url = baseUrl + link['href']

                    params = {'skip': 0, 'show': 2000}
                    sub_url_with_params = f"{sub_url}?{urlencode(params)}"

                    sub_response = requests.get(sub_url_with_params)

                    if sub_response.status_code == 200:
                        sub_soup = BeautifulSoup(sub_response.text, parser)

                        dl_tag = sub_soup.find('dl', {'id': 'articles'})

                        if dl_tag:
                            dt_tags = dl_tag.find_all('dt')

                            if dt_tags:
                                count = 0
                                for dt in dt_tags:
                                    a_tags = dt.find_all('a')
                                    for a in a_tags:
                                        if a.get('href') and 'html' in a['href']:
                                            count += 1

                                writer.writerow([link.text, sub_url_with_params, count])
                    else:
                        print(f"Error fetching sub-url: {sub_url_with_params}")
            else:
                print("Section not found!")
        else:
            print(f"Error: {response.status_code}")
