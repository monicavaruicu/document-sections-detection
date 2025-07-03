import os
import csv

current_directory = os.path.dirname(os.path.abspath(__file__))
labels_file = os.path.join(current_directory, 'tags.csv')

def shorten_label(label):
    words = label.strip().split()
    return (words[0][0] + words[1][0]).upper() if len(words) > 1 else words[0][0].upper()

def split_into_sections(input_file, output_file):
    tags = {}

    try:
        with open(labels_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row and len(row) == 3:
                    try:
                        weight, tag_name, alternatives = row
                        tags[tag_name.lower()] = {
                            'weight': int(weight),
                            'alternatives': [alt.strip().lower() for alt in alternatives.split(';')]
                        }
                    except ValueError:
                        print(f"Eroare la linia: {row}")
    except FileNotFoundError:
        print(f"Eroare: fișierul {labels_file} nu a fost găsit.")
        return


    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.readlines()
    except UnicodeDecodeError:
        with open(input_file, 'r', encoding='latin-1') as f:
            content = f.readlines()

    section_data = []
    in_section = False
    section = ""
    section_name = ""

    for line in content:
        line = line.strip()
        if line.startswith("Section:"):
            if in_section:
                section = ' '.join(section.split())
                matched = False
                for tag, tag_data in tags.items():
                    for alt in tag_data['alternatives']:
                        if alt and alt.strip() and alt in section_name.lower():
                            section_name = tag.strip().lower()
                            section_data.append((tag_data['weight'], section, section_name))
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    section_name = "Body"
                    section_data.append((100, section, "Body"))
            section_name = line[8:].strip().lower()
            section = ""
            in_section = True
        elif in_section:
            if not line.startswith("Section:"):
                section += "\n" + line

    if in_section:
        section = ' '.join(section.split())
        matched = False
        for tag, tag_data in tags.items():
            for alt in tag_data['alternatives']:
                if alt and alt.strip() and alt in section_name.lower():
                    section_name = tag.strip().lower()
                    section_data.append((tag_data['weight'], section, section_name))
                    matched = True
                    break
            if matched:
                break
        if not matched:
            section_name = "Body"
            section_data.append((100, section, "Body"))

    with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['paperName', 'sectionContent', 'sectionName'])
        section_data.sort(key=lambda x: x[0])
        for weight, section, name in section_data:
            name = name.title() if name != "Body" else "Body"
            short_name = shorten_label(name)
            writer.writerow([os.path.basename(input_file), section, short_name])
