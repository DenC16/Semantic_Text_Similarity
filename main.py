import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk                                                    
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

df = pd.read_csv('Precily_Text_Similarity.csv')
df.insert(0, 'ID', range(0, len(df)))

def countvectorizer_cosine_distance_method(s1, s2):
    
    allsentences = [s1 , s2]

    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)           
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cos_dist = cosine(text_to_vector_v1, text_to_vector_v2)
    return (1-cos_dist)

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

df.text1 = df.text1.apply(lambda x: preprocess(x))
df.text2 = df.text2.apply(lambda x: preprocess(x))

text1 = df.text1.tolist()
text2 = df.text2.tolist()

similarity_score = []
for index, row in df.iterrows():
  cosine_similarity = countvectorizer_cosine_distance_method(text1[index], text2[index])
  similarity_score.append(cosine_similarity)

Similarity_Score = [((x+1)/2) for x in similarity_score]

df = df.assign(Similarity_Score = Similarity_Score)
df = df[["ID","Similarity_Score"]]
df.to_csv("STS_score.csv")

