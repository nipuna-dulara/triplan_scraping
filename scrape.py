import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# URL of the page to scrape
base_url = 'https://www.holidify.com'
list_page_url = 'https://www.holidify.com/country/india/places-to-visit.html'
csv_file = 'places_data_india.csv'
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
else:
    existing_df = pd.DataFrame()
# Send a GET request to the list page
response = requests.get(list_page_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Create a list to store the extracted data
data = []

# Find all the divs with the required content
place_cards = soup.find_all('div', class_='content-card')

for card in place_cards:

    title = card.find('h3', class_='card-heading').get_text(strip=True)
    if not existing_df.empty and title in existing_df['Title'].values:
        print(f'Skipping {title} - already exists')
        continue
    print(f'Processing {title}')
    short_description = card.find('p', class_='card-text').get_text(strip=True)
    images = [img['data-original']
              for img in card.find_all('div', class_='lazyBG')]
    link = base_url + card.find('a')['href']

    # Download images
    image_folder = f'images_india/{title}'
    os.makedirs(image_folder, exist_ok=True)
    image_paths = []
    for idx, image_url in enumerate(images):
        image_response = requests.get(image_url)
        image_path = f'{image_folder}/{idx}.jpg'
        with open(image_path, 'wb') as img_file:
            img_file.write(image_response.content)
        image_paths.append(image_path)

    # Navigate to the detailed page
    detail_response = requests.get(link)
    detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
    # print(detail_soup)
    paragraphs = detail_soup.find_all(
        'div', class_='readMoreText')
    rating = detail_soup.find(
        'span', class_='rating-badge').get_text(strip=True)
    ideal_duration_p = detail_soup.find('b', text='Ideal duration: ')
    parent_p = ideal_duration_p.parent
    ideal_duration = parent_p.get_text(
        strip=True).replace('<b>Ideal duration: </b>', '')
    print(ideal_duration)
    # weather = detail_soup.find('p', text='Weather:').find_next(
    #     'span').get_text(strip=True)
    best_time_b = soup.find('b', text='Best Time: ')
    parent_p = best_time_b.parent
    best_time = parent_p.get_text(strip=True).replace(
        'Best Time: ', '').split('Read More')[0].strip()
    # ideal_duration = detail_soup.find('p', string="Ideal duration:")
    # print(ideal_duration)
    # you_must_know = soup.find(
    #     'div', class_='row no-gutters objective-information').get_text()
    faq = soup.find('div', class_='accordion').get_text()
    # best_time_b = soup.find('div', class_='accordion').get_text()
    if (len(soup.find_all('div', class_='col hfCarousel')) > 1):
        best_time_b = soup.find_all(
            'div', class_='col hfCarousel')[1].find_all('div', class_='card-body')
    elif (len(soup.find_all('div', class_='col hfCarousel')) > 0):
        best_time_b = soup.find_all(
            'div', class_='col hfCarousel')[0].find_all('div', class_='card-body')
    else:
        best_time_b = None
    # print(best_time_b)
    hotels = []
    if best_time_b:
        for hotel in best_time_b:
            hotels.append([hotel.find('div', class_="card-title").get_text(),
                           hotel.find('span', class_="price").get_text()])

    hotels = ", ".join([f"{hotel[0]}: {hotel[1]}" for hotel in hotels])
    thigs_to_do = soup.find(
        'div', class_='row no-gutters mb-50').find_all('div', class_='col-6 col-md-4 ptv-item')
    images = [img['data-original']
              for img in soup.find_all('img', class_='lazy')]
    things = ', '.join(
        [f"{thing.find('p').get_text()}" for thing in thigs_to_do])
    things_images = ', '.join([f"{img}" for img in images])
    ymk = ''
    if (len(paragraphs) > 1):
        ymk = paragraphs[1].get_text(separator=' ', strip=True)

    thisdf = pd.DataFrame([{
        'Title': title,
        'Short Description': short_description,
        'Images': ', '.join(image_paths),
        'Link': link,
        'Paragraphs': paragraphs[0].get_text(separator=' ', strip=True) or '',
        'Rating': rating,
        'Ideal Duration': ideal_duration,
        'Best Time': best_time,
        'You Must Know': ymk,
        'FAQ': faq,
        'Hotels': hotels,
        'Things to do': things,
        'Things to do images': things_images
    }])

    thisdf.to_csv(csv_file, mode='a', header=not os.path.exists(
        csv_file), index=False)
    print(f'Processed {title}')

# Save the data into a CSV file
df = pd.DataFrame(data)
# Assuming 'thisdf' is your DataFrame
df.to_csv('places_data_new.csv', mode='a', header=False, index=False)


print('Data saved to places_data.csv')
