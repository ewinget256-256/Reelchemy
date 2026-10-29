import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    gscores = pd.read_csv("data/genome-scores.csv")
    gtags = pd.read_csv("data/genome-tags.csv")
    return ratings, movies, gscores, gtags

ratings, movies, gscores, gtags = load_data()

@st.cache_data
def prep_data(ratings, movies, gscores):
    movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    genre_dummies = movies["genres"].str.get_dummies(sep="|")
    genome_scores = gscores.pivot(
        index = "movieId",
        columns = "tagId",
        values = "relevance"
        )
    genome_scores_matrix = genome_scores.loc[genome_scores.index.isin(movies["movieId"])]
    C = ratings["rating"].mean()
    return movie_stats, genre_dummies, genome_scores, genome_scores_matrix, C

movie_stats, genre_dummies, genome_scores, genome_scores_matrix, C = prep_data(ratings, movies, gscores)

movies["title"] = movies["title"].str.replace(
    r"^(.*), The (\(\d{4}\))$",
    r"The \1 \2",
    regex=True
)

st.title("Reelchemy")

movie_title = st.text_input("Search for a movie")
st.write("You searched for:", movie_title)

@st.cache_data
def recommend(movieId):

    def genre_rec(movie_id):

        given_movie_dummies = genre_dummies.loc[movies[movies["movieId"] == movie_id].index[0]]
        # calculate the cosine similarity between the given movie and all other movies and then flatten the result
        given_movie_scores = cosine_similarity(
            [given_movie_dummies],
            genre_dummies
        )
        given_movie_scores = given_movie_scores.flatten()
        # create a dataframe with the movieId, title, genres, and similarity scores for all movies
        given_movie_recs = movies[["movieId", "title", "genres"]].copy()
        given_movie_recs["similarity"] = given_movie_scores
        # merge the recommendations with the movie stats to get the mean and count of ratings for each recommended movie
        given_recs_with_ratings = pd.merge(given_movie_recs, movie_stats, on="movieId", how="left")
        #filter the recommendations to only include movies with at least 100 ratings
        given_recs_with_ratings = given_recs_with_ratings[given_recs_with_ratings["count"] >= 100]

        m = 1000
        v = given_recs_with_ratings["count"]
        R = given_recs_with_ratings["mean"]
        # calculate the weighted rating for each recommended movie using the IMDB formula and then calculate the combined score using the weighted rating and similarity score
        given_recs_with_ratings["weighted_rating"] = ((R * (v/(v+m))) + (C * (m/(v+m))))

        given_recs_with_ratings["normalized_rating"] = (given_recs_with_ratings["weighted_rating"] - 0.5) / (5 - 0.5)
        given_recs_with_ratings["cw_score"] = (given_recs_with_ratings["normalized_rating"] * 0.3) + (given_recs_with_ratings["similarity"] * 0.7)
        # sort the recommendations by the combined score and count of ratings in descending order and filter out the given movie from the recommendations
        given_recs_with_ratings = given_recs_with_ratings.sort_values(by=["cw_score", "count"], ascending=False)
        given_recs_with_ratings = given_recs_with_ratings[given_recs_with_ratings["movieId"] != movie_id]
        return given_recs_with_ratings
    
    if movieId not in genome_scores.index:
        return genre_rec(movieId)
    else:
        
        given_recs_with_ratings = genre_rec(movieId)

        given_mv_gvector = genome_scores.loc[movieId]

        movie_cosine_sim = cosine_similarity(
            [given_mv_gvector],
            genome_scores_matrix
        )

        movie_sim_scores = movie_cosine_sim.flatten()
        movie_sim_df = pd.DataFrame({
            "movieId": genome_scores_matrix.index,
            "similarity": movie_sim_scores
        })

        movie_sim_df = movie_sim_df[movie_sim_df["movieId"] != movieId]
        movie_sim_df = movie_sim_df.sort_values(by="similarity", ascending=False)
        movie_df = pd.merge(movie_sim_df, movies[["movieId", "title", "genres"]], on="movieId")
    
        hybrid_scores = pd.merge(
            given_recs_with_ratings[["movieId", "similarity"]].rename(columns={"similarity": "genre_similarity"}),
            movie_sim_df[["movieId", "similarity"]].rename(columns={"similarity": "genome_similarity"}),
    
            on="movieId",
            how = "left"
        )

        hybrid_scores["combined_similarity"] = (
            0.8 * hybrid_scores["genome_similarity"] +
            0.2 * hybrid_scores["genre_similarity"]
        )

        hybrid_scores["combined_similarity"] = np.where(hybrid_scores["genome_similarity"].isna(), hybrid_scores["genre_similarity"], hybrid_scores["combined_similarity"])
        hybrid_scores = hybrid_scores.sort_values("combined_similarity", ascending=False)

        titled_hybrid_scores = pd.merge(hybrid_scores, movies[["movieId", "title"]], on = "movieId", how = "left")
        return titled_hybrid_scores

if movie_title:
    results = movies[movies["title"].str.contains(movie_title, case=False, na=False)][["title", "genres"]]

    if not results.empty:
        selection = st.selectbox("Select a Movie", results["title"].tolist())

        selected_movie_id = movies.loc[
            movies["title"] == selection, "movieId"
        ].iloc[0]

        recommendations = recommend(selected_movie_id)
        st.dataframe(recommendations)
    else:
        st.write("no movies found")