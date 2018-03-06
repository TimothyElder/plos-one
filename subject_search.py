import time 
from functools import wraps #these two modules are for tracking the time in the proram 

start_time = time.time() #to measure the time elapsed in the program 

import os 
import pickle
import pandas as pd 
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import shutil # for copying the files. 

start_time = time.time() #to measure the time elapsed in the program 


def subject_extraction(soup_object):
	'''creates a function for finding a bunch of different social science articles, also takes all social science articles and copies them into a directory'''

	temp_subjects = []
	temp_subjects.append(soup_object.findAll('subject'))

	set_subjects = []
	subjects_temp = []
	for i in temp_subjects:
	    set_subjects=set(i)
	    subjects_temp.append(list(set_subjects))

	nested_list = []
	for nlist in subjects_temp:
	    row = []
	    for item in nlist:
	        row.append((item.text).lower())
	    nested_list.append(row)

	subjects = []
	for i in nested_list:
	    for thing in i:
	        subjects.append(thing)


	if 'psychology' in subjects:
		if 'correction' not in subjects:
			shutil.copy(file, psychology_articles_path)
			print('psychology article found')
	            
	if 'sociology' in subjects:
	    if 'correction' not in subjects:
	        shutil.copy(file, sociology_articles_path)
	        print('sociology article found')

	if 'linguistics' in subjects:
	    if 'correction' not in subjects:
	    	shutil.copy(file, linguistics_articles_path)
	    	print('linguistics article found')


	if 'anthropology' in subjects:
	    if 'correction' not in subjects:
	    	shutil.copy(file, anthropology_articles_path)
	    	print('anthropology article found')

	if 'social sciences' in subjects:
	    if 'correction' not in subjects:
	    	shutil.copy(file, social_science_articles_path)

psychology_articles_path = r'C:\Users\Timot\Documents\plos_organized_articles\psychology_articles'
sociology_articles_path = r'C:\Users\Timot\Documents\plos_organized_articles\sociology_articles'
linguistics_articles_path = r'C:\Users\Timot\Documents\plos_organized_articles\linguistics_articles'
anthropology_articles_path = r'C:\Users\Timot\Documents\plos_organized_articles\anthropology_articles'
economics_articles_path = r'C:\Users\Timot\Documents\plos_organized_articles\economics_articles'
social_science_articles_path = r'C:\Users\Timot\Documents\plos_organized_articles\social_science_articles'

print('Searching...')

target_dir = r'C:\Users\Timot\Documents\plos_data'
count = 0

for file in os.scandir(target_dir):
    with open(file, encoding="utf8") as f:
        soup = BeautifulSoup(f.read(), 'xml')
    
        subject_extraction(soup)


        count = count + 1

        if count == 5000:
        	print('5000 complete...')
        	end_time = time.time() -start_time 
        	print(end_time)

        if count == 10000:
        	print('10000 complete...')
        	end_time + time.time() - start_time
        	print(end_time)

        if count == 25000:
        	print('25000 complete...')
        	end_time = time.time() - start_time
        	print(end_time)

        if count == 50000:
        	print('50000 complete...')
        	end_time = time.time() - start_time
        	print(end_time)

        if count == 75000:
        	print('75000 complete...')
        	end_time = time.time() - start_time
        	print(end_time)

if count > 220000:
		    print('nearing completion...')


end_time = time.time() - start_time
print(end_time)