import os
import json
pages=100
categories = ['weapons-news-list','land-news-list','air-news-list','sea-news-list']
import requests
from bs4 import BeautifulSoup
import re
def fetch_articles(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    titleList=[]
    dateList=[]
    breifList=[]
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all div elements that match the class for the articles
        articles = soup.find_all('div', class_='list-item border-bottom border-primary px-3 pb-2 py-3')
        
        for article in articles:
            title_tag = article.find('h3')
            title = title_tag.text.strip() if title_tag else "No title found"
            titleList.append(title)
            link_tag = article.find('a', href=True)
            link = link_tag['href'] if link_tag else "No link found"

            # Extracting the news brief
            news_brief_tag = article.find('div', class_='overflow-hidden text-body')
            news_brief = news_brief_tag.text.strip().replace('\xa0', ' ') if news_brief_tag else "No news brief found"
            breifList.append(news_brief)
            # Extracting the timestamp
            date_tags = article.find_all('span', class_='pr-2 mb-2 tag text-muted')
            date_text = "No date found"
            for tag in date_tags:
                if re.search(r'\d{1,2} \w+ \d{4}', tag.text):
                    date_text = tag.text.strip()
                    break
            dateList.append(date_text)
            print(f" Date: {date_text}, News Brief: {news_brief}")
    else:
        print("Failed to retrieve the webpage")
        print(f"Status code: {response.status_code}")
    return titleList,dateList,breifList
def save_to_json(titles, dates, news_briefs, filename):
    # Check if all lists are of the same length
    if not (len(titles) == len(dates) == len(news_briefs)):
        raise ValueError("All lists must have the same length.")
    
    # Combine the lists into a list of dictionaries
    data = [{"title": title, "date": date, "newsBrief": brief} for title, date, brief in zip(titles, dates, news_briefs)]
    
    # Write the data to a JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
def file_exists(filename):
    return os.path.exists(filename)
def saveCategory(urlHead,cat,maxPage):
    titleList=[]
    dateList=[]
    breifList=[]
    for i in range(1,maxPage+1):
        fname="data/"+cat+"_page_"+str(i)+'.json'
        if (file_exists(fname)):
            print('skip ' + str(i) + " under " + cat)
        else:
            url=urlHead+cat+"/"
            t,d,b=fetch_articles(url)
            titleList=titleList+t
            dateList=dateList+d
            breifList=breifList+b
            os.system('touch '+fname)
            save_to_json(t,d,b,fname)
os.system('mkdir data')
for i in categories:
    saveCategory('https://www.janes.com/',i,pages)
