
# coding: utf-8

# In[ ]:


#Special module written for this class
#This provides access to data and to helper functions from previous weeks
#Make sure you update it before starting this notebook
import lucem_illud #pip install -U git+git://github.com/Computational-Content-Analysis-2018/lucem_illud.git

#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import nltk #For stop words and stemmers
import numpy as np #For arrays
import pandas as pd #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

#gensim uses a couple of deprecated features
#we can't do anything about them so lets ignore them 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning
get_ipython().run_line_magic('matplotlib', 'inline')

import os #For looking through files
import os.path #For managing file paths
from bs4 import BeautifulSoup
import pickle
import shutil
import lucem_illud #pip install git+git://github.com/Computational-Content-Analysis-2018/lucem_illud.git
import itertools
#All these packages need to be installed from pip
import requests #for http requests
import nltk #the Natural Language Toolkit
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import wordcloud #Makes word clouds
import numpy as np #For divergences/distances
import scipy #For divergences/distances
import seaborn as sns #makes our plots look nicer
import sklearn.manifold #For a manifold plot
from nltk.corpus import stopwords #For stopwords
import json #For API responses
import urllib.parse #For joining urls
import nltk
import csv

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# The first bit of code here is for taking a large set of xml documents from the AllofPLoS corpus and parsing them for important information. Here I am only taking 10,000 of the documents, and I am only extracting the article title, the journal title, the year it was published and the actual article contents. After the data is put into a dataframe I run a tokenizer and then a stemmer and save those additions to the original dataframe into their own dataframe. 

# In[2]:


#This cells navigates to the target directory and then opens each file individually and then passes it through beautifulsoup. This for loop will be the heart of the program I use to parse all the documents and extract the important information from them. 

target_dir = '../data/Data'
titles = []
journal_titles = []
file_names = []
bodies = []
years = []
text = []

count = 0


print("Searching Corpus...")

for file in (file for file in os.scandir(target_dir) if file.is_file() and not file.name.startswith('.')):
    file_names.append(file.name)
    with open(file.path) as f:

        soup = BeautifulSoup(f.read(), 'xml')
        
        b_temp = soup.body.find_all('p')
        for file in b_temp:
            text.append(file.text)
        bodies.append(''.join(text))
        text = []
        
        if soup.find('journal-title-group') == None:
            print('Error: no journal title')
            journal_titles.append('None')
        else:
            temp_journal = soup.find('journal-title')
            journal_titles.append(temp_journal.text)
    
        title_meta = soup.find('title-group')
        if title_meta.find_all('article-title') == '':
            print('Error: no title')
            titles.append('None')
        else:
            temp_titles = (title_meta.find('article-title'))
            titles.append(temp_titles.text)
        
        if soup.find('copyright-year') == None:
            print("Error: no copyright-year in file")
            years.append('None')
            print(file_names[count])
        else:
            temp_years = soup.find('copyright-year')
            years.append(temp_years.text)
    count = count+1
    if count == 1000:
        print('1000 complete...')
    if count == 5000:
        print('5000 complete...')
    if count == 9000:
        print('9000 complete...')
    if count == 10000:
        print('Corpus Search Complete. Have a nice day.')
        break


# In[3]:


with open("journal_list.txt", "wb") as fp:  #saves list
    pickle.dump(journal_titles, fp)

with open("years_list.txt", "wb") as fp:  #saves list
    pickle.dump(years, fp)

with open("titles_list.txt", "wb") as fp:  #saves list
    pickle.dump(titles, fp)

with open("bodies_list.txt", "wb") as fp:  #saves list
    pickle.dump(bodies, fp)
    
with open('file-names_list.txt', 'wb') as fp:
    pickle.dump(bodies, fp)


# In[4]:


with open("bodies_list.txt", "rb") as fp:
    bodies = pickle.load(fp)


# In[11]:


with open("titles_list.txt", "rb") as fp:
    titles = pickle.load(fp)

with open ("years_list.txt", "rb") as fp:
    years = pickle.load(fp)
    
with open("journal_list.txt", "rb") as fp:
    journal_titles = pickle.load(fp)


# In[2]:


plos_df = pandas.DataFrame({'Titles': titles, 'Article Contents': bodies, 'Journal Title' : journal_titles, 'Copyright Year' : years})


# In[19]:


plos_df.to_pickle('../data/plos_sample.pk1') #saves


# In[5]:


#loads dataframe
plos_df = pd.read_pickle('../data/plos_sample.pk1')
#plos_df


# In[ ]:


plosTokens = nltk.word_tokenize(plos_df['Article Contents'].sum())


# In[ ]:


count =0 
plosToken = []

print('Tokenizing...')
with open('tokenized_bodies.csv','w') as f1:
    writer=csv.writer(f1)
    for element in plos_df['Article Contents']:
        plosToken = nltk.word_tokenize(element)
        writer.writerow(plosToken)
        

#for element in plos_df['Article Contents']:
#    print(element)
#    plosToken.append(nltk.word_tokenize(element))
        count = count +1
        if count == 1000:
            print('1000')
        if count == 2000:
            print('2000')
        if count == 3000:
            print('3000')
        if count == 4000:
            print('4000')
        if count == 5000:
            print('5000')
        if count == 6000:
            print('6000')
        if count == 7000:
            print('7000')
        if count == 8000:
            print('8000')
        if count == 9000:
            print('9000')
        if count == 10000:
            print('10000 tokenized. Keep It Clean.')
            break
    


# In[5]:


tokenized = csv.reader(open("tokenized_bodies.csv","r"))


# In[ ]:


tokenized_bodies = []

print("Appending list...")
for row in tokenized:
    tokenized_bodies.append(row)


# In[3]:


with open('tokenized-text_list.txt', 'wb') as fp:
    pickle.dump(tokenized_bodies, fp)


# In[42]:


with open('tokenized-text_list.txt', 'wb') as fp:
    pickle.dump(tokenized_bodies, fp)


# In[43]:


#two days later I think this is all the text tokenized. 
plosTokens = flat_list


# In[44]:


#stems and normalizes the text. appending the data frame with them

plos_df['tokenized_text'] = plosTokens

plos_df['word_counts'] = plos_df['tokenized_text'].apply(lambda x: len(x))


# In[ ]:


#stems and normalizes the text. appending the data frame with them. plos_df['tokenized_text] is each document tokenized, so
#for me to replicate this I will have to go back and tokenize each document and create a list of those tokenized documents. 

plos_df['tokenized_text'] = plos_df['Article Contents'].apply(lambda x: nltk.word_tokenize(x))

plos_df['word_counts'] = plos_df['tokenized_text'].apply(lambda x: len(x))


# ## Corpus Linguistics
# With the data cleaned and saved, the following code starts to do some rudimentary analysis of the corpus, suuch as word frequencies and parts of speech tags. 

# In[46]:


countsDict = {}
for word in plosTokens:
    if word in countsDict:
        countsDict[word] += 1
    else:
        countsDict[word] = 1
word_counts = sorted(countsDict.items(), key = lambda x : x[1], reverse = True)
word_counts[:100]


# In[48]:


#The stop list is then all words that occur before the first noun
#for the corpus as it is I think that the stop words should start with the noun 'disease'
stop_words_freq = []
for word, count in word_counts:
    if word == 'data':
        break
    else:
        stop_words_freq.append(word)
stop_words_freq
wordnet = nltk.stem.WordNetLemmatizer()


# In[49]:


#the following is the function that actually normalizes the tokens
stop_words_nltk = stopwords.words('english')
#stop_words = ["the","it","she","he", "a"] #Uncomment this line if you want to use your own list of stopwords.

#The stemmers and lemmers need to be initialized before bing run
porter = nltk.stem.porter.PorterStemmer()
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()

def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)
        
    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)
    
    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
    #We will return a list with the stopwords removed
    return list(workingIter)


#this part of the code might look similar to what is done before but is actually different. 
#The previous code twas simply tokenizing the data, this is normalizing those tokens.
plos_df['normalized_tokens'] = plos_df['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = porter))

plos_df['normalized_tokens_count'] = plos_df['normalized_tokens'].apply(lambda x: len(x))

plos_df


# In[22]:


#The data is now cleaned, tokenized and normalized. Saving dataframe so in the future I don't have to rerun all the lines above. 
#Remember to keep the two data frames seperate though. 
plos_df.to_pickle('../data/plos_normalized_sample.pk1') 


# In[23]:


#The following will be substantive analysis 
#.sum() adds together the lists from each row into a single list
plos_cfdist = nltk.ConditionalFreqDist(((len(w), w) for w in plos_df['normalized_tokens'].sum()))

#print the number of words
print(plos_cfdist.N())


# In[37]:


#distribution of different word lengths. the '.plot()' method provides a simple line graph but with so much data it is just an opaque group of words. 
plos_cfdist[8]


# In[38]:


wc = wordcloud.WordCloud(background_color="white", max_words=500, width= 1000, height = 1000, mode ='RGBA', scale=.5).generate(' '.join(plos_df['normalized_tokens'].sum()))
plt.imshow(wc)
plt.axis("off")


# In[39]:


#the following cells have code that find meaningful collocations, starting with bigrams.
plosBigrams = nltk.collocations.BigramCollocationFinder.from_words(plos_df['normalized_tokens'].sum())
print("There are {} bigrams in the finder".format(plosBigrams.N))


# In[43]:


#shows raw counts of 
def bigramScoring(count, wordsTuple, total):
    return count

print(plosBigrams.nbest(bigramScoring, 20))


# In[45]:


print(plosBigrams.nbest(bigramScoring, 40))


# # Clustering and Topic Modeling

# In[3]:


#load the dataframe with the normalized tokens appended.
plos_df = pd.read_pickle('../data/plos_normalized_sample.pk1')


# In[5]:


#This turns documents into word count vectors.
#The question is whether or not to use the raw words, or to use the tokenized words, or the stemmed words. 
#First it needs to be initialized
plosCountVectorizer = sklearn.feature_extraction.text.CountVectorizer()
#Then trained
plosVects = plosCountVectorizer.fit_transform(plos_df['Article Contents'])
print(plosVects.shape)


# In[8]:


plosVects


# In[32]:


#gets the indices of the word in the parantheses. Note that the value this gives is not a frequency, but a number associated with the words position in the sparse matrix
plosCountVectorizer.vocabulary_.get('science')


# In[33]:


#this allows for the use of tf-idf, which allows for document distinguishing, while excluding highly frequent words which are less meaningful in distinguishing between documents
#initialize
plosTFTransformer = sklearn.feature_extraction.text.TfidfTransformer().fit(plosVects)
#train
plosTF = plosTFTransformer.transform(plosVects)
print(plosTF.shape)


# In[37]:


#gives the tf-idf of each word
list(zip(plosCountVectorizer.vocabulary_.keys(), plosTF.data))[:10]
#for the values given for each word, the smaller the value the less informative that word is in the corpus, informative to what specifically I think is the ability to distinguish different documetns. 


# In[38]:


#This 'prunes' the matrix, meaning it only leaves certain words based upon frequency and weighting, I think. 
#initialize
plosTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, stop_words='english', norm='l2')
#train
plosTFVects = plosTFVectorizer.fit_transform(plos_df['Article Contents'])


# In[49]:


plosVects


# In[51]:


#comparing this and the above cell it is easy to see what the tf-idf pruning is doing. It is limiting what words are going to be analyzed by their weights.
#Now there are 1000 words but the same amount of documents. 
plosTFVects


# In[65]:


#similar as to what was done above this allows us to see what a words indices are in the matrix, but thematrix is so much smaller now that the value given will be much smaller than the possible ones above. 
try:
    print(plosTFVectorizer.vocabulary_['certain'])
except KeyError:
    print('vector is missing')
    print('The available words are: {} ...'.format(list(plosTFVectorizer.vocabulary_.keys())[:]))


# In[69]:


#similar as to what was done above this allows us to see what a words indices are in the matrix, but thematrix is so much smaller now that the value given will be much smaller than the possible ones above. 
try:
    print(plosTFVectorizer.vocabulary_['pain'])
except KeyError:
    print('vector is missing')
    print('The available words are: {} ...'.format(list(plosTFVectorizer.vocabulary_.keys())[:20]))


# The truncated matrix I just created using the TF-IDF weights is a great place to being identifying clusters

# ## Flat Clustering
# The first method I am going to work with requires that I set the number of clusters that the documents can fit into. In the homework example this was reasonable because the documents were drawn from a particular set of pre-existing categories based on topic. For my corpus and the sake of this exercise I will use the same number of clusters as the homework example, this might be mathodolgically unsond or meaningless. Hopefully there is an unsupervised method which will allow a natural number of clusters to arise. 

# In[74]:


numClusters = 4


# In[75]:


#initaalizes cluster finder
km = sklearn.cluster.KMeans(n_clusters=numClusters, init='k-means++')


# In[76]:


#this calculates the clusters
km.fit(plosTFVects)


# In[83]:


#This evaluates the calculated clusters using 4 metrics, 

print("Evaluating Clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(plos_df['Journal Title'], km.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(plos_df['Journal Title'], km.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(plos_df['Journal Title'], km.labels_)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(plos_df['Journal Title'], km.labels_)))
#I dont' think that any of these values actually mean anything because the numebr of clsuters I picked has no real meaning. And where plos_df['Journal Title] is above can be several different things and I don't know which is the most appropriate.


# In[84]:


#assigns cluster predictions to dataframe. But I don't think this method actually makes any sense becaue my choice of number of clusters is meaningless.
plos_df['kmeans_predictions'] = km.labels_
plos_df

