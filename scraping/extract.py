
from urllib.parse import urlparse
import requests
from langchain_community.docstore.document import Document as LangChainDocument
from bs4 import BeautifulSoup
from docx import Document as WordDocument
import re
import os

file_path = r'C:\Users\HP\Desktop\New ML\MultiRAG-scraping\scraping\pages_name.txt'

lists = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        url = line.strip()
        if url:
            lists.append(url)


def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove scripts and styles
    for element in soup(['script', 'style', 'noscript']):
        element.decompose()

    return soup.get_text(separator=' ', strip=True)


def fetch_webpage(url):
    response = requests.get(
        url,
        timeout=20,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    response.raise_for_status()
    return response.text


def clean_text(text):
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove unwanted characters
    text = re.sub(r"""[^a-zA-Z0-9\s.,!?'"()-]""", '', text)

    return text


def write_to_docx(text_content, output_filename):
    doc = WordDocument()

    doc.add_heading('Document Title', level=1)
    doc.add_paragraph(text_content)

    doc.save(output_filename)


for url in lists:
    try:
        print(f"Scraping: {url}")

        html_content = fetch_webpage(url)
        page_text = parse_html(html_content)
        final_text = clean_text(page_text)

        # Generate a safe filename
        parsed_url = urlparse(url)
        filename = parsed_url.path.rstrip('/').split('/')[-1]

        if not filename:
            filename = parsed_url.netloc

        filename = re.sub(r'[^\w.-]', '_', filename)

        output_filename = filename + '.docx'

        write_to_docx(final_text, output_filename)

        print(f"Saved: {output_filename}")

    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")

    except Exception as e:
        print(f"Error processing {url}: {e}")

print("Scraping completed!")
