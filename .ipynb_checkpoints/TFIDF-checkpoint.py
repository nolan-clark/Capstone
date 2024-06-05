# TF-IDF Pipeline

from utils import tokenize_stems
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# DATA
df = pd.read_csv('') # Dataset with 'text' column

# Set thresholds for TF-IDF of bigrams and trigrams
MIN = 0.02
MAX = 1.0

# Stem and tokenize
txt1 = tokenize_stems(df)

# TF-IDF vectorizer
vectorizer = CountVectorizer(ngram_range=(2,3), min_df=MIN, max_df=MAX
                            )
X1 = vectorizer.fit_transform(txt1)
features = vectorizer.get_feature_names()

vectorizer = TfidfVectorizer(ngram_range=(2,3), min_df=MIN, max_df=MAX
                            )
X2 = vectorizer.fit_transform(txt1)
scores = X2.toarray()

# Preview output
sums = X2.sum(axis=0)
data1 = []

for col, term in enumerate(features):
    data1.append((term, sums[0,col]))

ranking = pd.DataFrame(data1, columns = ['term','rank'])
words = ranking.sort_values('rank', ascending = False)

print(words.shape)

# TF-IDF Dataframe 
tfidf_df = pd.DataFrame(X2.toarray(),columns=vectorizer.get_feature_names())

tfidf_df.to_csv('',index=False) # output location