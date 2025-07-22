import csv
import os
import requests
from bs4 import BeautifulSoup

txts_directory = './TXTs'
parser = 'html.parser'

def extract_text_content(url, title, sections_file):
    if not os.path.exists(txts_directory):
        os.makedirs(txts_directory)
    
    file_path = os.path.join(txts_directory, f"{title}.txt")
    
    response = requests.get(url)

    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, parser)

        authors_tag = soup.find_all(class_='ltx_authors')
        for tag in authors_tag:
            tag.decompose()

        svg_tag = soup.find_all(class_='ltx_picture')
        for tag in svg_tag:
            tag.decompose()

        keywords_tag = soup.find_all(class_='ltx_keywords')
        for tag in keywords_tag:
            tag.decompose()

        navbar_tag = soup.find_all(class_='ltx_page_navbar')
        for tag in navbar_tag:
            tag.decompose()

        bibliography_tag = soup.find_all(class_='ltx_bibliography')
        for tag in bibliography_tag:
            tag.decompose()

        footer_tag = soup.find_all(class_='ltx_page_footer')
        for tag in footer_tag:
            tag.decompose()

        item_tag = soup.find_all(class_='ltx_tag ltx_tag_item')
        for tag in item_tag:
            tag.decompose()

        section_number_tag = soup.find_all(class_='ltx_tag ltx_tag_section')
        for tag in section_number_tag:
            tag.decompose()

        for figure_tag in soup.find_all('figure'):
            figure_tag.decompose()

        subsection_title_tag = soup.find_all(class_='ltx_title ltx_title_subsection')
        for tag in subsection_title_tag:
            tag.decompose()

        subsubsection_title_tag = soup.find_all(class_='ltx_title ltx_title_subsubsection')
        for tag in subsubsection_title_tag:
            tag.decompose()

        abstract_tag = soup.find_all(class_='ltx_abstract')
        for tag in abstract_tag:
            tag.decompose()

        title_tag = soup.find_all(class_='ltx_title ltx_title_document')
        for tag in title_tag:
            tag.decompose()

        appendix_tag = soup.find_all(class_='ltx_appendix')
        for tag in appendix_tag:
            tag.decompose()

        title_paragraph_tag = soup.find_all(class_='ltx_title ltx_title_paragraph')
        for tag in title_paragraph_tag:
            tag.decompose()

        ref_title_tag = soup.find_all(class_='ltx_text ltx_ref_title')
        for tag in ref_title_tag:
            tag.decompose()

        theorem_tag = soup.find_all(class_='ltx_tag ltx_tag_theorem')
        for tag in theorem_tag:
            tag.decompose()

        for math_tag in soup.find_all('math'):
            alttext = math_tag.get('alttext')
            if alttext:
                math_tag.string = alttext

        title_section_tags = soup.find_all(class_='ltx_title ltx_title_section')
        section_titles = [tag.get_text(strip=True) for tag in title_section_tags]

        acknowledgements_tags = soup.find_all(class_='ltx_title ltx_title_acknowledgements')
        section_titles.extend(tag.get_text(strip=True) for tag in acknowledgements_tags)

        with open(sections_file, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([title, ', '.join(section_titles)])

        section_tags = soup.find_all(class_='ltx_title ltx_title_section')
        for tag in section_tags:
            tag.string = "Section: " + tag.get_text(strip=True)

        acknowledgements_tags = soup.find_all(class_='ltx_title ltx_title_acknowledgements')
        for tag in acknowledgements_tags:
                tag.string = "Section: " + tag.get_text(strip=True)


        text_content = soup.get_text()

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_content)
    else:
        print("Error:", response.status_code)