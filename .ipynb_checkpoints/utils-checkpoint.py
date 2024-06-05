# Util functions for the pipeline

# Normalize Text Function -- remove punctuation, lowercasing, etc. for TF-IDF tokenization
import re

def remove_noise(string):
    
    stripped_string = re.sub('[^0-9a-zA-Z ]+', ' ', string).lower()
    stripped_string = re.sub('\s+',' ',stripped_string)
    
    stripped_string = stripped_string.strip()
    
    if stripped_string !='':
        return stripped_string

# Stem and Tokenize Text
def tokenize_stems(data):
    #data.reset_index(inplace=True)
    txt1 = []
    for i in data['text']:
        txt1.append(remove_noise(i))

    # remove stop words and stem
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = PorterStemmer()
    for i, essay in enumerate(txt1):
        txt1[i] = ' '.join([
            ps.stem(x) for x in nltk.word_tokenize(essay) if x not in stop_words])
    return txt1

# Cosine Similarity Duplicate Removal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer

def cos_sim_filter(data, threshold=0.85):
    data.reset_index(inplace=True)
    txt1 = []
    for i in data['text']:
        txt1.append(remove_noise(i))

    # remove stop words and stem
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = PorterStemmer()
    for i, essay in enumerate(txt1):
        txt1[i] = ' '.join([
            ps.stem(x) for x in nltk.word_tokenize(essay) if x not in stop_words])
    # cosine_similarity
    vectorizer = TfidfVectorizer()
    vector1= vectorizer.fit_transform(txt1)
    cosine_matrix = cosine_similarity(vector1)

    # Set diagonal to 0 to ignore self-similarity
    np.fill_diagonal(cosine_matrix, 0)
    
    # Find duplicate pairs
    duplicates = np.where(cosine_matrix >= threshold)
    
    # Create a set to track indices to remove
    to_remove = set()
    for i, j in zip(duplicates[0], duplicates[1]):
        if i < j:  # Ensure each pair is considered only once
            to_remove.add(j)
    
    # Remove duplicates from the DataFrame
    df_unique = data.drop(to_remove)
    print('Removed Duplicates:', len(data) - len(df_purged))
    
    return df_purged
    
# Outlier Removal - spelling errors
from tqdm import tqdm
from spellchecker import SpellChecker
def remove_spelling_outliers(data):
    txt1 = []
    for i in data['text']:
        txt1.append(remove_noise(i))
    
    bucket = {}
    for i,j in tqdm(enumerate(txt1)):
        spell = SpellChecker()
        misspelled = spell.unknown(j.split())
        bucket[i]=len(misspelled)

    miss_counts=pd.DataFrame(bucket.values())
    miss_counts.columns=['word_count']
    ceiling=miss_counts['word_count'].quantile(q=0.99)
    df_filtered=data[miss_counts['word_count'] <=ceiling].reset_index(drop=True)

    print('Removed Outliers: ',(df['label'].value_counts() - df_filtered['label'].value_counts()), sep='\n')

    return df_filtered
    

# Holm's Correction -- only include significant variables
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def correct_holm(humans,llms):
    # dictionary of variables : p-values 
    bucket = {}
    # conduct unparied t-test for each variable
    for variable in humans:
        t_stat, p_value = ttest_ind(humans[variable], llms[variable])
        bucket[variable] = p_value
    
    # bonferroni correct of p_values [0]=reject [1]=p-values corrected
    bonf=multipletests(list(bucket.values()),alpha=0.01,method='holm')
    
    # collect significant variables
    sig_corrected_variables = []
    for i,j in enumerate(bonf[0]):
        if j:
            sig_corrected_variables.append(list(bucket.keys())[i])
    
    # return significant variable names, significant corrected pvals
    return sig_corrected_variables, bonf[1][bonf[0]]

