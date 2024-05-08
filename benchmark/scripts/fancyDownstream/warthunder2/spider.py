import requests
from bs4 import BeautifulSoup


def extract_tech_tree_fighters(url="https://wiki.warthunder.com/Category:USA_aircraft"):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        fighter_list = []
        link_list = []
        # Find all the fighter names and descriptions
        fighter_items = soup.find_all('div', class_='tree-item')

        for item in fighter_items:
            fighter_name = item.find('a').text.strip()
            fighter_link = item.find('a')['href']
            link_list.append('https://wiki.warthunder.com' + fighter_link)
            print(fighter_link)
            description = "N.A"
            fighter_list.append(fighter_link[1:])

        return fighter_list, link_list
    else:
        print("Failed to retrieve data from the website.")
        return None


def extract_plane_description(plane_url):
    response = requests.get(plane_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the <h2> tag with class 'mw-headline' and ID 'Description'
        description_heading = soup.find('span', {'class': 'mw-headline', 'id': 'Description'})
        if description_heading:
            # Find the following <p> tag for the description
            description_paragraph = description_heading.find_next('p')
            if description_paragraph:
                # Extract the text of the paragraph
                description = description_paragraph.text.strip()
                return description
            else:
                print("Description paragraph not found.")
                return 'N.A.'
        else:
            print("Description heading not found.")
            return 'N.A.'
    else:
        print("Failed to retrieve data from the website.")
        return None


def extract_plane_usage(plane_url):
    response = requests.get(plane_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the <h2> tag with class 'mw-headline' and ID 'Description'
        description_heading = soup.find('span', {'class': 'mw-headline', 'id': 'Usage_in_battles'})
        if description_heading:
            # Find the following <p> tag for the description
            description_paragraph = description_heading.find_next('p')
            if description_paragraph:
                # Extract the text of the paragraph
                description = description_paragraph.text.strip()
                return description
            else:
                print("Description paragraph not found.")
                return 'N.A.'
        else:
            print("Description heading not found.")
            return 'N.A.'
    else:
        print("Failed to retrieve data from the website.")
        return None


def extract_plane_history(plane_url):
    response = requests.get(plane_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the <h2> tag with class 'mw-headline' and ID 'Description'
        description_heading = soup.find('span', {'class': 'mw-headline', 'id': 'History'})
        if description_heading:
            # Find the following <p> tag for the description
            description_paragraph = description_heading.find_next('p')
            if description_paragraph:
                # Extract the text of the paragraph
                description = description_paragraph.text.strip()
                return description
            else:
                print("Description paragraph not found.")
                return 'N.A.'
        else:
            print("Description heading not found.")
            return 'N.A.'
    else:
        print("Failed to retrieve data from the website.")
        return None


def append_str_to_file(filename, string):
    with open(filename, 'a') as file:
        file.write(string + '\n')


import os


def file_exists(filename):
    return os.path.exists(filename)


import numpy as np


# get the description of planes from War Thunder
def getDescriptionCtx(maxParase=-1, fname='USA_aircraft'):
    urlHead = "https://wiki.warthunder.com/Category:"
    fighter_list, link_list = extract_tech_tree_fighters(urlHead + fname)
    ru = []
    cnt = 0
    os.system('mkdir progress')
    os.system('mkdir data')
    idx = 0
    os.system("mkdir data/" + fname)
    for ti in range(len(link_list)):
        i = link_list[ti]
        j = fighter_list[ti]
        idx = idx + 1
        strI = str(idx)
        progressFname = "progress/" + fname + "_" + strI
        if (file_exists(progressFname)):
            print('skip ' + j + " under" + fname)
        else:
            tstr = "Description of " + j + ": \n" + extract_plane_description(i)
            append_str_to_file("data/" + fname + "/keys" + ".txt", j)
            append_str_to_file("data/" + fname + "/description_" + strI + ".txt", tstr)
            usageStr = "Usage of " + j + ": \n" + extract_plane_usage(i)
            append_str_to_file("data/" + fname + "/usage_" + strI + ".txt", usageStr)
            historyStr = "History of " + j + ": \n" + extract_plane_history(i)
            append_str_to_file("data/" + fname + "/history_" + strI + ".txt", historyStr)
            ru.append(tstr)
            print(tstr)
            os.system('touch ' + progressFname)
        if (maxParase > 0):
            if (cnt == maxParase):
                return ru
            cnt = cnt + 1
    return ru


def load_list_from_file(filename):
    with open(filename, 'r') as file:
        string_list = file.readlines()
        # Remove trailing newline characters
        string_list = [string.strip() for string in string_list]
        return string_list


if __name__ == "__main__":
    countries = [
        'USA',
        'Germany',
        'USSR',
        'Britain',
        'Japan',
        'China',
        'Italy',
        'France',
        'France',
        'Sweden',
        'Israel'
    ]
    types = [
        'aircraft',
        'ground_vehicles'
    ]
    categories = []
    for i in countries:
        for j in types:
            tempStr = i + '_' + j
            categories.append(tempStr)
    print(categories)
    for i in categories:
        getDescriptionCtx(-1, i)
