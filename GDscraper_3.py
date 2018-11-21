# coding: utf-8

# In[1]:


from splinter import Browser
from bs4 import BeautifulSoup
from random import randint
import datetime
import os
import json
import time
import re

# Sets the root url
root_url = 'https://www.glassdoor.com'

# Sets the search parameters
time_period = "Last Day"
keyword = 'Data Scientist'
location = 'United States'
# Set path variables

corpus_path = 'corpus_master/'



# Sets the path to the chrome driver
executable_path = {'executable_path': '/usr/local/bin/chromedriver'}
browser = Browser('chrome', **executable_path, headless=False)

# directs the browser to visit the root site and sleeps for 2 seconds
browser.visit(root_url)
time.sleep(2)

# Fills out the form for job title and location and sleeps for 2 seconds
browser.find_by_xpath("//input[@id='KeywordSearch']").fill(keyword)
browser.find_by_xpath("//input[@id='LocationSearch']").fill(location)
browser.find_by_id('HeroSearchButton').click()
time.sleep(2)

# Select date posted drop down and Set date range to search
browser.find_by_text("Date Posted").click()
time.sleep(1)
browser.find_by_text(time_period).click()
time.sleep(randint(1,3))
  


# In[ ]:


# initialize page count
n = 1

# run while there is still a next button
while True:
    try:
        x_button = browser.find_by_css('div.xBtn')
        if x_button:
            x_button.click()
    except Exception as e:
        print(e)
    test_url = browser.url
    
    
    #creates directory to save all files
    date = datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y')
    path = corpus_path + date + '/'
    if not os.path.exists(path):
        os.makedirs(path) 

    
    
    elements = browser.find_by_xpath("//li[@class='jl']")
    
    for element in elements:
        try:
            x_button = browser.find_by_css('div.xBtn')
            if x_button:
                x_button.click()
        except Exception as e:
            print(e)
        try:
            element.click()
            time.sleep(2)
            job_title = browser.find_by_xpath("//h1[@class='jobTitle h2 strong']")

            #print(job_title.html)


            job_title = job_title.html

            job_title = str(job_title)

            company = str(browser.find_by_xpath("//a[@class='plain strong empDetailsLink']").html)
            print(company)

            company_info = browser.find_by_xpath("//div[@class='compInfo']")

            if len(company_info.find_by_tag('span')) == 2:
                city_state = str(company_info.find_by_tag('span')[1].html)
                rating = str(company_info.find_by_tag('span')[0].html)
                rating = rating[0:3]
                print(city_state)

                print(rating)
            elif len(company_info.find_by_tag('span')) == 1:
                city_state = company_info.find_by_tag('span').html
                rating = 'NA'
                print(city_state)
                print(rating)

            job_description = str(browser.find_by_xpath("//div[@class='jobDescriptionContent desc']").html)

            date = datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y')

            job = {'job title': job_title, 'company': company, 'city state': city_state, 'rating': rating, 'job description': job_description}

            try:
                filename = company.replace(" ", "") + '-' + job_title.replace(" ", "").replace("/", "-") + '.json'
                with open(path + filename, 'w') as f:
                    f.write(json.dumps(job))
                    print('Saved to file : ' + filename)
            except Exception as e:
                print(e)
            time.sleep(2)
        except Exception as e:
            print(e)
    
    next_button = browser.find_by_css('li.next')
    if next_button:
        next_button.click()
        time.sleep(randint(1,10))
    else:
        break
    n += 1

