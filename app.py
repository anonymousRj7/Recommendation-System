import streamlit as st
import pickle


st.header("Movie Recommendation System")

movie_cosine_similarity= pickle.load(open('similarity.pkl', 'rb'))
df = pickle.load(open('movies_df.pkl', 'rb'))

def recommendation(movie_name):
  movie_idx = df[df['title'] == movie_name].index[0]
  sim_scores = list(enumerate(movie_cosine_similarity[movie_idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:11]
  sim_scores
  movie_indices = [i[0] for i in sim_scores]
  recommendations = df['title'].iloc[movie_indices].tolist()

  return recommendations


if st.checkbox("Show all Movies name : "):
   st.write(df)

movie_name  = st.text_input("Enter the movie name : ")


if st.button("Get Recommendation"):
  if movie_name :
    with st.spinner("Loading recommendation..."):
        recommendations = recommendation(movie_name)
        st.write("Top recommendations based on user's historical movie preferences:")
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")

