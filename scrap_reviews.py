import sys, os, getopt
import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
import pandas as pd
#import queue
import re
import time
from selenium import webdriver
#from selenium.common.exceptions import TimeoutException

#TODO: Replace python time.sleep(5) with webdriver's wait_untill.

def filter_links(links_list, sub_string_list):
    regex = '|'.join("%s" % re.escape(s) for s in sub_string_list)
    return [link for link in links_list if re.search(regex, link)]


def get_all_pages_html(browser, web_urls, break_at_link_no = 100, break_at_review_page_no = 200):
    print("Downloading the HTML data from URLs")
    page_html_list = []
#    browser = webdriver.Firefox(executable_path=r"C:\Users\Adubey4\Desktop\anand_nlp\geckodriver.exe")
    for url_indx, web_url in enumerate(web_urls):
        if url_indx >=break_at_link_no:
            break;
        browser.get(web_url)
        review_page_count = 0
        while True:
            review_page_count +=1
            if review_page_count>=break_at_review_page_no:
                break;
            time.sleep(5)
            a = browser.find_element_by_css_selector(".partial_entry .taLnk.ulBlueLinks")
            a.click()
            time.sleep(5)
            # sleep vs storage: use bs or not
            page_html_list.append([web_url,bs(browser.page_source, "lxml").findAll("div", { "class" : "wrap" })])
            next_tag = browser.find_element_by_css_selector(".nav.next")
            if next_tag.is_enabled():
                next_tag.click()
            else:
                break;
    return page_html_list;
#web_url = "https://www.tripadvisor.in/Hotel_Review-g297654-d8625588-Reviews-Conrad_Pune-Pune_Pune_District_Maharashtra.html"
#get_all_pages_html([web_url])


def get_all_review_links(browser, base_web_url, break_at_page_no=100):
    print("Getting urls for hotel reviews")
#    browser = webdriver.Firefox(executable_path=r"C:\Users\Adubey4\Desktop\anand_nlp\geckodriver.exe")
    browser.get(base_web_url)    
    next_tag = browser.find_element_by_css_selector(".nav.next.taLnk.ui_button.primary")
    i = 0    
    all_links = []    
    while True:
        if i >= break_at_page_no:
            break;
        time.sleep(5)
        print("waiting ...")
        next_tag = browser.find_element_by_css_selector(".nav.next.taLnk.ui_button.primary")
        new_i = int(next_tag.get_attribute("data-page-number"))
        print(new_i)
        if new_i > i:
            i = new_i
            all_anchors = browser.find_elements_by_css_selector('a')
            all_links += [urljoin(base_web_url, a_tag.get_attribute('href')) for a_tag in all_anchors if a_tag.get_attribute('href') ]
        if next_tag.is_enabled():
            next_tag.click()
        else:
            break;
    return all_links

def extract_data_from_tags(web_reviews_html):
    print("Getting data from HTM reviews")
    all_reviews = []
    for page_url ,bs_html_tag_list in web_reviews_html:
        for bs_html_tag in bs_html_tag_list:
            title_tag = bs_html_tag.find("span", { "class" : "noQuotes" })
            title = None
            review = None
            if title_tag:
                title = title_tag.text
            review_tag = bs_html_tag.find("p", { "class" : "partial_entry" })
            if review_tag:
                review = review_tag.text
            if title or review:
                all_reviews.append([page_url, title, review])
    return all_reviews;

# http://htmlformatter.com/
if __name__ == "__main__":
    browser = webdriver.Firefox(executable_path=r"C:\Users\Adubey4\Desktop\anand_nlp\geckodriver.exe")
    all_hotels_links = []
    web_reviews_html = []
    all_reviews = []
    base_web_url = "https://www.tripadvisor.in/Hotels-g297654-Pune_Pune_District_Maharashtra-Hotels.html"
    all_hotel_review_links = get_all_review_links(browser, base_web_url,3)
    all_hotel_review_links = filter_links(all_hotel_review_links, ["Hotel_Review-"])
    web_reviews_html = get_all_pages_html(browser, all_hotel_review_links, 2,3)
    all_reviews = extract_data_from_tags(web_reviews_html)
    browser.close()
