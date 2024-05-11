import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle



df = pd.read_csv("movies.csv")


df = df.iloc[:10000,:] #Taking Small Subset of Dataset
df['genres'] = df['genres'].str.replace('|', ' ')
df['genres'] = df['genres'].str.lower()
df['title'] = df['title'].str.extract(r'(^.+)\s\(\d{4}\)')  # To remove the year from movie name
df['title'] = df['title'].str.split(',').str[0] # To correct the movie name



tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['genres'])


# print(tfidf_vectorizer.get_feature_names_out())
# print(tfidf_matrix.shape)


movie_cosine_similarity = cosine_similarity(tfidf_matrix)
# print(movie_cosine_similarity.shape)


pickle.dump(df, open('movies_df.pkl', 'wb'))
pickle.dump(movie_cosine_similarity, open('similarity.pkl', 'wb'))
print(f"File are dumped")