# Real-Time Mapping oF Depression Using Twitter (Bitcamp 2019) 

## Problem 
Major depression is one of the most common mental health issues around the globe. Oftentimes, those who are depressed go to social media such as Twitter and Facebook to express their feelings. Oftentimes, these signs are left undetected or they are not handled properly. Our novel approach aims to solve this challenge by mapping tweets indicative of depression as well as crisis centers across the United States. This is a fast and efficient way of finding localized areas of high depression and providing crisis centers with information as to how prevalent depression is in their area. 

## Solution 
We first utilized the Twint tool for real-time scraping of Twitter data in the United States during the past two days. We then utilized NLTK tools to preprocess the text from the tweets and transform it into data that could be fed into a pre-trained Convolutional LSTM network. Once we filtered tweets were indicative of depression, we used the Twint tool to access a general location of the user and used ArcGIS geo-encoding API to extract longitude and latitude coordinates. Our web application, which relies on a Flask framework then took these coordinates and utilized the Google Maps API to create a heatmap. We were also able to extract locations of crisis centers from the Suicide Prevention Hotline website and plot those on our map. After creating this, we focused on making it real-time, so that the heatmap can be updated based on new tweets. 

