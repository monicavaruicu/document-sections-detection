import os
import requests
from bs4 import BeautifulSoup

parser = 'html.parser'
htmls_directory = './HTMLs'

def extract_html_content(url, title):
        if not os.path.exists(htmls_directory):
            os.makedirs(htmls_directory)

        file_path = os.path.join(htmls_directory, f"{title}.html")

        response = requests.get(url)

        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, parser)

            body = soup.body.prettify()
            html_content = [body]
            
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("\n".join(html_content))
        else:
            print("Error:", response.status_code)