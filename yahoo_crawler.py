import Date
import time
import requests
import format_text
from bs4 import BeautifulSoup


DELIMITER = ""
RETRY_PERIOD = 60
BASE_URL = "https://finance.yahoo.com/q/h?s="


def news_spider(stock_ticker_symbol, date_from, date_to):
    """
    Given the stock ticker symbol and the date range within which the article
    should have been published, returns the textual content of all such articles
    """
    text_list = []  # initialize a text variable to store all the news articles
    is_incomplete = False
    current_date = Date.Date(date_from)  # create the necessary Date objects
    target_date = Date.Date(date_to)
    while current_date.is_earlier(target_date) and not is_incomplete:  # as long as the Date is within the range
        # create the specific url and there is not yet a single date that does not have news articles
        try:
            url = BASE_URL + stock_ticker_symbol + "&t=" + current_date.url_format()
            source_code = requests.get(url)  # visit the website and get the source code
            plain_text = source_code.text  # convert the source code into plain text
            # store all the code into a Beautiful Soup object
            soup = BeautifulSoup(plain_text, "html.parser")
            # get the part of the source code that contains the news
            news_box = soup.find("div", attrs={"class": "mod yfi_quote_headline withsky"})
            h3 = news_box.find("h3")  # find the first date heading of the page
            # if the current date is the date in the heading
            if current_date.news_heading_format() in h3.string:
                print(h3.string)
                print("Crawling...")
                date_ul = news_box.find("ul")  # find the box of links to the news articles
                news_ul = date_ul.find_next("ul")
                # for every link in the list
                for link in news_ul.findAll("a"):
                    # extract the href which is essentially the link
                    href = str(link.get("href")).split("*")[-1]
                    # extract all the text in that website and store it
                    text_list.append(extract_text(href))
            else:
                is_incomplete = True  # if the date is not in the heading, there must be missing news
            current_date.advance_date()  # go to the next day
        except Exception as error_message:  # handle connection errors
            print(error_message)
            print("Connection Error - Retrying in " + str(RETRY_PERIOD) + " seconds...")
            time.sleep(RETRY_PERIOD)  # retry in 60 seconds if faced with a connection error
    if is_incomplete:  # if there is missing news
        if len(text_list) > 0:  # if there is news so far
            final_text = text_list[0]  # obtain the news associated with the first date only
        else:  # if there is no news
            final_text = ""  # store the null string as the final text to be returned
    else:  # if the news is complete
        # return the text of all the news articles crawled so far
        final_text = format_text.join_words(text_list, DELIMITER)
    return final_text, is_incomplete  # return the final text and the status of the news, i.e. complete/incomplete


def extract_text(url):
    """
    Given an url, visits the website, extracts and returns all the text in the
    website
    """
    text = ""  # initialize a text variable to store all the text
    source_code = requests.get(url)  # visit the website and get the source code
    plain_text = source_code.text  # convert the source code into plain text
    # store all the code into a Beautiful Soup object
    soup = BeautifulSoup(plain_text, "html.parser")
    # for every paragraph on the web page
    for paragraph in soup.findAll("p"):
        # convert the NavigableString object to a string object and store it
        text += str(paragraph.text)
    return text  # return all the text from the web page crawled


