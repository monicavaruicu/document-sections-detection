import re

def split_into_sentences(text):
    
    abbreviations = r'\b(?:[Ff]ig|[Ee]x|[Ee]g|[Dd]r|[Ii]ng|[Ee]t al|[Ee]ng)\.'
    text = re.sub(abbreviations, lambda m: m.group(0).replace('.', '<PERIOD>'), text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<PERIOD>', '.') for s in sentences]

    return sentences
