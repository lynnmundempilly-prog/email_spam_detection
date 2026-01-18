from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(texts):
    vectorizer=TfidfVectorizer(stop_words='english')#✔ Tokenization,✔ Stopword removal,✔ TF-IDF weighting
    X=vectorizer.fit_transform(texts)#Learns the vocabulary,Calculates IDF values,Converts each email into a numeric vector
    return X,vectorizer