import json

def load_from_json(filename):
    # Read data from the JSON file
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data into separate lists
    titles = [item['title'] for item in data]
    dates = [item['date'] for item in data]
    news_briefs = [item['newsBrief'] for item in data]

    return titles, dates, news_briefs
def loadCategory(cat,maxPage,prefix='janes/'):
    titleList=[]
    dateList=[]
    breifList=[]
    for i in range(1,maxPage+1):
        fname=prefix+"data/"+cat+"_page_"+str(i)+'.json'
        t,d,b=load_from_json(fname)
        titleList=titleList+t
        dateList=dateList+d
        breifList=breifList+b
    return titleList,dateList,breifList
def loadCategories(cats,maxPage,prefix='janes/'):
    titleList=[]
    dateList=[]
    breifList=[]
    for i in cats:
        t,d,b=loadCategory(i,maxPage,prefix)
        titleList=titleList+t
        dateList=dateList+d
        breifList=breifList+b
    return titleList,dateList,breifList

import random

def mixSentences(briefings, num_mixes=10):
    """
    Randomly mix sentences from different briefings to create incorrect contexts.

    :param briefings: List of strings, where each string is a briefing containing one or more sentences.
    :param num_mixes: Number of mixed briefing contexts to generate.
    :return: List of strings, each a mixed context briefing.
    """
    all_sentences = []
    # Split each briefing into sentences and collect them into a single list
    for briefing in briefings:
        sentences = briefing.split('. ')
        all_sentences.extend([sentence.strip() for sentence in sentences if sentence])

    mixed_contexts = []
    for _ in range(num_mixes):
        mixed_context = ' '.join(random.sample(all_sentences, k=min(5, len(all_sentences))))
        mixed_contexts.append(mixed_context)

    return mixed_contexts
import re
def modifyNumbers(briefings):
    """
    Detect all numbers in the briefings and multiply them by 0.5.

    :param briefings: List of strings, where each string is a briefing containing text and numbers.
    :return: List of strings, each a briefing with numbers modified.
    """
    modified_briefings = []
    number_pattern = r'\b\d+\.?\d*\b'  # Regex to match integers and decimals

    for briefing in briefings:
        # Replace each number found with its half value
        modified_briefing = re.sub(number_pattern, lambda match: str(float(match.group()) * 0.5), briefing)
        modified_briefings.append(modified_briefing)

    return modified_briefings