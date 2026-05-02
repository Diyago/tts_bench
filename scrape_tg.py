import requests
from bs4 import BeautifulSoup

url = 'https://t.me/s/ai_tablet'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

messages = soup.find_all('div', class_='tgme_widget_message_text')
for i, msg in enumerate(messages):
    print(f"--- Post {i} ---")
    print(msg.get_text(separator='\n'))
