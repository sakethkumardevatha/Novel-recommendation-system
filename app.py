import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/novels_cleaned.csv")
    df.fillna("", inplace=True)
    return df

# TF-IDF & cosine similarity
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Recommend novels
def recommend(title, df, cosine_sim):
    indices = pd.Series(df.index, index=df['title'].str.lower())
    title = title.lower()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    novel_indices = [i[0] for i in sim_scores]
    return df[['title', 'authors', 'genres']].iloc[novel_indices]

# Streamlit UI
st.title("📚 Novel Recommendation System")

df = load_data()
cosine_sim = build_model(df)

st.sidebar.header("🔎 Search or Filter")

search_query = st.sidebar.text_input("Search novel by title")
genres_list = sorted(set(g.strip("[]'\" ") for sublist in df['genres'].str.split(',') for g in sublist))
selected_genres = st.sidebar.multiselect("Filter by genres", genres_list)

uploaded_file = st.sidebar.file_uploader("Upload your own dataset", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.fillna("", inplace=True)
    cosine_sim = build_model(df)
    st.success("✅ Custom dataset uploaded successfully!")

# Search result
if search_query:
    results = df[df['title'].str.contains(search_query, case=False, na=False)]
    if not results.empty:
        st.subheader("🔍 Search Results")
        st.dataframe(results[['title', 'authors', 'genres']].head(10))
    else:
        st.warning("No novels found!")

# Genre filtering
if selected_genres:
    genre_filter = df[df['genres'].apply(lambda x: all(genre.lower() in x.lower() for genre in selected_genres))]
    if not genre_filter.empty:
        st.subheader("🎯 Filtered Novels by Genre")
        st.dataframe(genre_filter[['title', 'authors', 'genres']].head(10))
    else:
        st.warning("No novels match selected genres.")

# Recommendation
st.subheader("🤖 Get Recommendations")
novel_input = st.text_input("Enter a novel title you liked:")
if novel_input:
    recs = recommend(novel_input, df, cosine_sim)
    if not recs.empty:
        st.write("📖 You may also like:")
        st.dataframe(recs)
    else:
        st.error("Title not found in dataset. Try a different one.")
