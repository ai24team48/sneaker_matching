import bs4
import requests
import logging
import collections
import csv
import os
from urllib.request import urlretrieve
from playwright.sync_api import sync_playwright


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('poison')


ParseResult = collections.namedtuple(
    'ParseResult',
    (
        'url',
        'brand',
        'model',
        'color',
        'characteristics',
        'gender',
        'price',
        'pictures_url',
        'pictures_in_file',
    )
)
HEADERS = (
    'Ссылка',
    'Бренд',
    'Модель',
    'Цвет',
    'Характеристики',
    'Пол',
    'Цена',
    'Картинки',
    'Картинки_в_файле',
)

class Client:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Accept-Language' : 'ru',
        }
        self.result = []

    def load_page(self, url_ , page: int = None):
    #    with sync_playwright() as p:
    #        browser = p.chromium.launch(headless=False)
    #        page_1 = browser.new_page()
    #        page_1.goto("https://poizoncom.ru/r-muzhskie-krossovki-two?filter_brands=14649,15403,1120443,1162156&filter_price_to=21155")
#
    #        page_1.wait_for_selector('body')
    #        page_1.evaluate("window.scrollTo(0, document.body.scrollHeight);")
     #       url = f'https://poizoncom.ru/r-muzhskie-krossovki-two?filter_brands=14649,15403,1120443,1162156&filter_price_to=21155?page={page}'
     #       res = self.session.get(url=url)
     #       return res.text

        if page is None:
            page = 1

        url = f"{url_}{page}"

        logger.info(url)
        #url = f'https://poizoncom.ru/r-muzhskie-krossovki-two?page={page}'
        #url = f"https://poizoncom.ru/brand-nike?page={page}"
        #url = f'https://poizoncom.ru/r-muzhskie-krossovki-two?filter_brands=14649,15403,1120443,1162156&filter_price_to=21155?page={page}'
        res = self.session.get(url=url)
        res.raise_for_status() # Проверка успешности запроса
        return res.text

    def download_image(self, image_url, image_path):
        # Выполняем запрос с заголовком User-Agent
        headers = {
            'User -Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            with open(image_path, 'wb') as out_file:
                out_file.write(response.content)
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")

    def parse_page(self, text: str):
        soap = bs4.BeautifulSoup(text, 'lxml')
        container = soap.select('div.product-listing-card')
        #logger.info('%s', container )
        for block in container:
            self.parse_block(block = block)

    def parse_block(self, block):
        #logger.info(block)
    ###
        url_block = block.select_one('a.product-listing-card__preview-link')
        if not url_block:
            logger.error('no url_block')
            return

        url = url_block.get('href')
        if not url:
            logger.error('no href')
            return
        logger.info('%s', url)

        try:
            product_page = requests.get(f'https://poizoncom.ru{url}')
            product_page.raise_for_status()  # Проверяем статус ответа

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP ошибка: {http_err}")  # Обработка ошибок HTTP
            return
        product_page.raise_for_status()  # Проверка успешности запроса
        product_soup = bs4.BeautifulSoup(product_page.text,'lxml')
        #image_element = product_soup.find('img', class_='preview-image__img')
    #    folder_path = "downloaded_images"
       # img_tags = product_soup.find_all('a', class_='product-preview-carousel__wrapper')
    #    div = product_soup.find('div', class_='product-preview-carousel__wrapper')
    #    img_urls = [a_tag['href'] for a_tag in div.find_all('a', href=True)]
    #    logger.info(div)
    #    logger.info('%s', img_urls )
    #    image_names = []
    #    for image_url in img_urls:
    #        try:
    #            if image_url:
    #                # Загрузка изображения
    #                image_filename = os.path.basename(image_url)
    #                image_names.append(image_filename)
    #                image_path = os.path.join('poison_images', image_filename)
    #                os.makedirs('poison_images', exist_ok=True)  # Создание директории для изображений
    #                self.download_image(image_url, image_path)
    #                logger.info('Downloaded image: %s', image_path)
    #            else:
    #                logger.error('No image URL found')
    #        except Exception as e:
    #            print(f"Ошибка при загрузке изображения {image_url}: {e}")

    ###
        image_element = product_soup.find('img', class_='preview-image__img')
        if image_element:
            image_url = image_element.get('src')
            if image_url:
                # Загрузка изображения
                image_filename = os.path.basename(image_url)
                image_path = os.path.join('images', image_filename)
                os.makedirs('images', exist_ok=True)  # Создание директории для изображений
                self.download_image(image_url, image_path)
                #urlretrieve(image_url, image_path)
                logger.info('Downloaded image: %s', image_path)
            else:
               logger.error('No image URL found')
        else:
            logger.error('No image element found')

        info_block = block.select_one('div.product-listing-card-info')
        if not info_block:
            logger.error('no info_block')
            return
        brand = info_block.get('title')
        logger.info('%s', brand)

        model_block = product_soup.select_one('div.card-product-layout__area-title')
        if not model_block:
            logger.error('no model_block')
            return
        model = model_block.find('h1', id='name').get_text(strip=True)
        logger.info('%s', model)

        description_block = product_soup.select_one('div.collapsed-block__inner')
        logger.info('%s', description_block)
        if not description_block:
            logger.error('no description_block')
            return
        gender_element = description_block.find('span', text='Пол')
        if gender_element:
            gender = gender_element.find_parent('dt').find_next_sibling('dd').get_text(strip=True)
        else:
            gender = None
        logger.info('%s', gender)
        color_element = description_block.find('span', text='Цвет')
        if color_element:
            color = color_element.find_parent('dt').find_next_sibling('dd').get_text(strip=True)
        else:
            color = None
        logger.info('%s', color)

        text_block = product_soup.select_one('div.card-product-layout__block-description')
        logger.info('%s', text_block)

        characteristics = {}
        if text_block.get_text(strip=True) and text_block.find_all('ul'):
            for li in text_block.find_all('ul')[0].find_all('li'):
                key_tag = li.find(['strong', 'b'])
                if key_tag:
                    key = key_tag.get_text(strip=True)[:-1]
                    value = li.get_text(strip=True).replace(key + ': ', '')
                    characteristics[key] = value
        logger.info('%s', characteristics)

        price_block = block.select_one('span.product-listing-card-prices__price')
        if not price_block:
            logger.error('no price_block')
            return
        price = price_block.text.strip().replace('₽', '').replace('&nbsp;', '').replace(' ', '')
        logger.info('%s', price)



        self.result.append(ParseResult(
            url = url,
            brand = brand,
            model = model,
            color = color,
            characteristics = characteristics,
            gender = gender,
            price = price,
            pictures_url = image_url,
            pictures_in_file = image_filename,
        ))

    def save_results(self):
        path = 'C:\\Users\\Пользователь\\PycharmProjects\\pythonProject9\\data.csv'
        with open(path, 'w' , encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting = csv.QUOTE_MINIMAL)
            writer.writerow(HEADERS)
            for item in self.result:
                writer.writerow(item)


    def run(self):
        urls = ['https://poizoncom.ru/brand-nike?page=', 'https://poizoncom.ru/brand-adidas?page=', 'https://poizoncom.ru/brand-new-balance?page=']
        for url in urls:
            for i in range(1, 11):
                text = self.load_page(url, i)
                self.parse_page(text = text)
                logger.info(f'Итого: {len(self.result)}')
        self.save_results()


if __name__ == '__main__':
    parser = Client()
    parser.run()