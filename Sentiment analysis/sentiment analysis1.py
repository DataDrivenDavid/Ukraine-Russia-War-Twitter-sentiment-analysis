#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


import pandas as pd

# Correcting the file path and using raw string literal
file_path = r'C:/Users/sirve/Downloads/filename.csv'

# Load the dataset
df= pd.read_csv(file_path)

df.head()


# In[9]:


df.columns


# In[10]:


df.info()


# In[12]:


# List of important columns to keep
important_columns = ['tweet', 'language', 'replies_count', 'retweets_count', 'likes_count', 'hashtags']

# Drop all other columns
data= df[important_columns]

# Display the first few rows of the cleaned data
data.head()


# In[13]:


data.isnull().sum()


# In[ ]:





# In[ ]:





# ## Exploratory data analysis

# #### Language distribution 

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

# Language Distribution
language_distribution = data['language'].value_counts()
#print(language_distribution)

# Plot Language Distribution (optional, for verification)
plt.figure(figsize=(10, 6))
language_distribution.plot(kind='bar')
plt.title('Distribution of Tweets by Language')
plt.xlabel('Language')
plt.ylabel('Number of Tweets')
plt.show()


# #### Tweet length distribution 

# In[35]:


import warnings
warnings.filterwarnings('ignore')

# Tweet Length Distribution
data['tweet_length'] = data['tweet'].apply(len)
print(data['tweet_length'].describe())

# Plot Tweet Length Distribution (optional, for verification)
plt.figure(figsize=(10, 6))
data['tweet_length'].plot(kind='hist', bins=50)
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()


# #### Most Common Words and Hashtags:

# In[16]:


from collections import Counter
import re

# Common words
all_words = ' '.join(data['tweet']).lower()
all_words = re.findall(r'\b\w+\b', all_words)
common_words = Counter(all_words).most_common(20)

# Plot common words
words, counts = zip(*common_words)
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.title('Most Common Words in Tweets')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Common hashtags
all_hashtags = data['hashtags'].sum()
common_hashtags = Counter(all_hashtags).most_common(20)

# Plot common hashtags
hashtags, counts = zip(*common_hashtags)
plt.figure(figsize=(10, 6))
plt.bar(hashtags, counts)
plt.title('Most Common Hashtags in Tweets')
plt.xlabel('Hashtags')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# #### User Engagement 

# In[17]:


# User Engagement
print(data[['replies_count', 'retweets_count', 'likes_count']].describe())

# Plot User Engagement (optional, for verification)
plt.figure(figsize=(10, 6))
data[['replies_count', 'retweets_count', 'likes_count']].plot(kind='box')
plt.title('Distribution of Replies, Retweets, and Likes')
plt.ylabel('Count')
plt.show()


# ### Sentiment analysis

# In[19]:


#pip install textblob


# In[26]:


import re
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')


# Extract the 'tweet' column
tweets = data['tweet'].dropna()

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    return text

tweets_cleaned = tweets.apply(preprocess_text)

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis using TextBlob
sentiment_scores_textblob = tweets_cleaned.apply(analyze_sentiment_textblob)

# Classify sentiment based on polarity score
def classify_sentiment_textblob(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

sentiment_classification_textblob = sentiment_scores_textblob.apply(classify_sentiment_textblob)

# Add the results to the original dataframe
data['cleaned_tweet'] = tweets_cleaned
data['sentiment_score'] = sentiment_scores_textblob
data['sentiment'] = sentiment_classification_textblob

# Display the first few rows of the updated dataframe
print(data[['tweet', 'cleaned_tweet', 'sentiment_score', 'sentiment']].head())


# In[27]:


# Function to create and display a bar plot for each sentiment category
def plot_sentiment(sentiment):
    sentiment_data = data[data['sentiment'] == sentiment]
    sentiment_counts = sentiment_data['sentiment'].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='virAidis')
    plt.title(f'{sentiment} Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()


# In[28]:


#Plot the sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[ ]:


#pip install wordcloud


# In[31]:


from wordcloud import WordCloud

def plot_wordcloud(sentiment):
    sentiment_data = data[data['sentiment'] == sentiment]
    text = ' '.join(sentiment_data['cleaned_tweet'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{sentiment} Sentiment Word Cloud')
    plt.axis('off')
    plt.show()


# In[32]:


plot_wordcloud('Positive')


# In[33]:


plot_wordcloud('Negative')


# In[34]:


plot_wordcloud('Neutral')


# In[ ]:




