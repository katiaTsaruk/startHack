import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from plyer import notification
import requests
from bs4 import BeautifulSoup
import re

import tkinter as tk
from tkinter import messagebox

import pandas as pd
from transformers import pipeline
from urllib.parse import urlparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import webbrowser


# Initialize the BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
YOUR_API_KEY = "f579b73fdd5d2768b4f4ff635dd347fd"  # Replace with your GNews API key

# Predefined lists of countries and regions
COUNTRIES = ["USA", "Germany", "China", "Japan", "India", "France", "UK", "Canada", "Turkey"]
REGIONS = {
    "North America": ["USA", "Canada", "Mexico"],
    "Europe": ["Germany", "France", "UK", "Italy", "Spain", "Turkey"],
    "Asia": ["China", "Japan", "India", "South Korea"],
}

class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground='blue', underline=1)
        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)
        self.reset()

    def reset(self):
        self.links = {}

    def add(self, action):
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event):
        self.text.config(cursor="hand2")

    def _leave(self, event):
        self.text.config(cursor="")

    def _click(self, event):
        for tag in self.text.tag_names(tk.CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return


def show_popup_notification(title: str, message: str, url: str):
    popup = tk.Tk()
    popup.title(title)

    text = tk.Text(popup)
    text.pack()

    hyperlink = HyperlinkManager(text)

    def click_action():
        webbrowser.open(url)

    text.insert(tk.INSERT, message, hyperlink.add(click_action))

    button = tk.Button(popup, text="OK", command=popup.destroy)
    button.pack()

    popup.mainloop()


def check_articles_for_japan_tk(df):
    japan_articles = df[df["country"] == "Japan"]
    if not japan_articles.empty:
        for index, article in japan_articles.iterrows():
            title = f"New article about {article['keyword']} in Japan!"
            message = f"There is a new article about {article['keyword']} in Japan. Here is the link: "
            # we separate text "summary" into a different line due to tkinter Text widget behaviour of applying tags
            summary = f"\n\n Summary: \n {article['summary']}"
            link = article['link']
            # show pop-up notification
            show_popup_notification(title, message, link)
            show_popup_notification(title, summary, link)


def send_email_notification(subject: str, msg_body: str, recipient_email: str, your_email: str, your_password: str):
    msg = MIMEMultipart()
    msg['From'] = your_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(msg_body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    server.login(your_email, your_password)
    text = msg.as_string()
    server.sendmail(your_email, recipient_email, text)
    server.quit()


def check_articles_for_japan(df):
    japan_articles = df[df["country"] == "Japan"]
    if not japan_articles.empty:
        for index, article in japan_articles.iterrows():
            subject = f"New article about {article['keyword']} in Japan!"
            message = "There is a new article about {} in Japan. Here is the link: {} \n The summary is \n {}".format(
                article["keyword"], article["link"], article["summary"])
            send_email_notification(subject, message, 'oek0974@thi.de', 'oek0974@thi.de', '')



"""def show_popup_notification(title: str, message: str):
    popup = tk.Tk()
    popup.withdraw()  # Hides the main tkinter window
    messagebox.showinfo(title, message)
    popup.destroy()  # Destroys the tkinter window

def check_articles_for_japan_tk(df):
    japan_articles = df[df["country"] == "Japan"]
    if not japan_articles.empty:
        for index, article in japan_articles.iterrows():
            title = f"New article about {article['keyword']} in Japan!"
            message = f"There is a new article about {article['keyword']} in Japan. Here is the link: {article['link']} \n\n Summary: \n{article['summary']}"
            # show pop-up notification
            show_popup_notification(title, message)"""

def clean_text(text):
    #Clean the input text by removing special characters and extra spaces.
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()  # Convert to lowercase


def extract_region_from_url(url):
    #Extract region information from the URL.
    domain = urlparse(url).netloc.lower()  # Get the domain from the URL

    if 'eu' in domain:
        return "Europe"
    elif 'us' in domain or 'com' in domain:
        return "North America"
    elif 'uk' in domain:
        return "Europe"  # Assuming UK falls under Europe for now
    elif 'ca' in domain:
        return "North America"  # Canada
    elif 'cn' in domain:
        return "Asia"  # China
    elif 'jp' in domain:
        return "Asia"  # Japan
    elif 'in' in domain:
        return "Asia"  # India
    elif 'tr' in domain:
        return "Europe"  # Turkey
    # Additional country/region logic can be added here
    else:
        return "Unknown"


def detect_region_country(url, content):
    #Detect the country and region based on the URL and article content.
    # First, try to extract region/country from the URL
    detected_region = extract_region_from_url(url)

    # Then, try to extract the country and region from the content (if needed)
    detected_country = "Unknown"
    for country in COUNTRIES:
        if country.lower() in content.lower():
            detected_country = country
            break

    # Try to detect the region from the country if not found from URL
    if detected_country == "Unknown" and detected_region != "Unknown":
        for region, countries in REGIONS.items():
            if detected_region in region:
                detected_country = countries[0]  # Get the first country in the region

    return detected_country, detected_region


def analyze_trend(headline, article_text):
    """Generate trend analysis summary based on headline and article text."""
    # Implementation of trend analysis summarization using BART model pipeline


def get_articles(keyword, num_articles=5):
    """Retrieves news articles via the GNews API."""
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": keyword,
        "max": num_articles,
        "lang": "en",
        "token": YOUR_API_KEY,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()["articles"]

def analyze_trend(headline, article_text):
    #Generate a trend analysis summary based on the headline and article text.
    # Check if content is None or empty then return appropriate response
    if not article_text:
        return "Article content is not available to generate a summary"

    input_text = headline + " " + article_text
    input_length = len(input_text.split())

    # Dynamically set max and min length for summarization
    max_len = min(100, input_length)  # Set max length for summarization
    min_len = max(20, max_len - 30)  # Ensure min_len is smaller than max_len, but not below 5

    if input_length < 20:
        min_len = 1  # If the input text is too short, adjust min_len to 1

    try:
        # Summarize with updated length constraints
        summary = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Trend analysis could not be generated due to: {str(e)}"



def scrape_and_analyze_articles(keywords, num_articles=5):
    articles = []
    for keyword in keywords:
        articles_data = get_articles(keyword, num_articles)
        for article in articles_data:
            headline = article.get("title", "No Title")
            full_link = article.get("url", "No Link")
            article_text = article.get("description", "")

            country, region = detect_region_country(full_link, article_text)
            trend_summary = analyze_trend(headline, article_text)
            articles.append({
                "keyword": keyword,
                "headline": headline,
                "link": full_link,
                "country": country,
                "region": region,
                "summary": trend_summary,
                "clean_headline": clean_text(headline),
            })

    return pd.DataFrame(articles)


# Keywords to search for
keywords = ["automotive batteries", "industrial batteries", "consumer batteries"]

# Scrape and analyze GNews articles for the keywords
df = scrape_and_analyze_articles(keywords)

# Drop columns 'headline' and 'clean_headline'
df = df.drop(columns=["headline", "clean_headline"], errors="ignore")

# Display the trend analysis summaries
print(df[["keyword", "summary", "link", "country", "region"]])

# Check articles for Japan and send notification
#check_articles_for_japan(df)
check_articles_for_japan_tk(df)

# Optionally, save the summary to an Excel file
df.to_excel("battery_trends_summary_gnews.xlsx", index=False)

"""
# Initialize the BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Predefined lists of countries and regions, including China, Japan, India, and Turkey
COUNTRIES = ["USA", "Germany", "China", "Japan", "India", "France", "UK", "Canada", "Turkey"]
REGIONS = {
    "North America": ["USA", "Canada", "Mexico"],
    "Europe": ["Germany", "France", "UK", "Italy", "Spain", "Turkey"],
    "Asia": ["China", "Japan", "India", "South Korea"],
}


def clean_text(text):
    #Clean the input text by removing special characters and extra spaces.
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()  # Convert to lowercase


def extract_region_from_url(url):
    #Extract region information from the URL.
    domain = urlparse(url).netloc.lower()  # Get the domain from the URL

    if 'eu' in domain:
        return "Europe"
    elif 'us' in domain or 'com' in domain:
        return "North America"
    elif 'uk' in domain:
        return "Europe"  # Assuming UK falls under Europe for now
    elif 'ca' in domain:
        return "North America"  # Canada
    elif 'cn' in domain:
        return "Asia"  # China
    elif 'jp' in domain:
        return "Asia"  # Japan
    elif 'in' in domain:
        return "Asia"  # India
    elif 'tr' in domain:
        return "Europe"  # Turkey
    # Additional country/region logic can be added here
    else:
        return "Unknown"


def detect_region_country(url, content):
    #Detect the country and region based on the URL and article content.
    # First, try to extract region/country from the URL
    detected_region = extract_region_from_url(url)

    # Then, try to extract the country and region from the content (if needed)
    detected_country = "Unknown"
    for country in COUNTRIES:
        if country.lower() in content.lower():
            detected_country = country
            break

    # Try to detect the region from the country if not found from URL
    if detected_country == "Unknown" and detected_region != "Unknown":
        for region, countries in REGIONS.items():
            if detected_region in region:
                detected_country = countries[0]  # Get the first country in the region

    return detected_country, detected_region

def analyze_trend(headline, article_text):
    #Generate a trend analysis summary based on the headline and article text.
    # Check if content is None or empty then return appropriate response
    if not article_text:
        return "Article content is not available to generate a summary"

    input_text = headline + " " + article_text
    input_length = len(input_text.split())

    # Dynamically set max and min length for summarization
    max_len = min(100, input_length)  # Set max length for summarization
    min_len = max(20, max_len - 30)  # Ensure min_len is smaller than max_len, but not below 5

    if input_length < 20:
        min_len = 1  # If the input text is too short, adjust min_len to 1

    try:
        # Summarize with updated length constraints
        summary = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Trend analysis could not be generated due to: {str(e)}"


def scrape_articles(keywords, num_articles=5):
    #Scrape articles based on keywords and generate trend analysis.
    base_url = "https://news.google.com"
    articles = []

    for keyword in keywords:
        search_url = f"{base_url}/search"
        params = {"q": keyword, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        response = requests.get(search_url, params=params)

        if response.status_code != 200:
            print(f"Failed to fetch articles for keyword: {keyword}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.find_all("article")[:num_articles]:
            # Extract headline
            headline_tag = article.find("h3")
            headline = headline_tag.text if headline_tag else "No Title"

            # Extract link
            link_tag = article.find("a")
            relative_link = link_tag["href"] if link_tag else None
            full_link = f"{base_url}{relative_link[1:]}" if relative_link else "No Link"

            # Scrape the linked news article for detailed content
            article_text = ""
            if full_link != "No Link":
                try:
                    article_response = requests.get(full_link)
                    if article_response.status_code == 200:
                        article_soup = BeautifulSoup(article_response.text, "html.parser")
                        paragraphs = article_soup.find_all("p")
                        if paragraphs:
                            article_text = ' '.join([p.text for p in paragraphs])
                        else:
                            print(f"No text content could be found in the webpage at URL: {full_link}")
                    else:
                        print(f"Failed to fetch article at URL: {full_link}")
                except Exception as e:
                    print(f"Error fetching article content: {e}")

            # Detect region and country based on the URL and article text
            country, region = detect_region_country(full_link, article_text)

            # Generate trend analysis summary
            trend_summary = analyze_trend(headline, article_text)

            # Append result to articles list
            articles.append({
                "keyword": keyword,
                "headline": headline,
                "link": full_link,
                "country": country,
                "region": region,
                "summary": trend_summary,
                "clean_headline": clean_text(headline),
            })

    return pd.DataFrame(articles)



# Keywords to search for
keywords = ["automotive batteries", "industrial batteries", "consumer batteries"]

# Scrape and analyze articles
df = scrape_articles(keywords)

# Drop 'headline' and 'clean_headline' columns
df = df.drop(columns=["headline"], errors="ignore")  # Ignore if 'headline' does not exist
df = df.drop(columns=["clean_headline"], errors="ignore")  # Ignore if 'clean_headline' does not exist


# Show the first few rows of the result
print(df[["keyword", "summary", "link", "country", "region"]])

# Optionally, save to an Excel file
df.to_excel("battery_trends_summary_updated.xlsx", index=False)
"""