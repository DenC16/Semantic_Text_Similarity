import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk                                               
nltk.download('stopwords')
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



def preprocess(text, stem=True):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

def countvectorizer_cosine_distance_method(s1, s2):

    # sentences to list
    allsentences = [s1, s2]
    changed_sentence = list(map(preprocess, allsentences))

    # text to vector
    vectorizer = CountVectorizer()
    # Vectorization through Bag of Words method
    all_sentences_to_vector = vectorizer.fit_transform(changed_sentence)
    
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()

    # distance of similarity
    cos_dist = cosine(text_to_vector_v1, text_to_vector_v2)
    return (1-cos_dist)
