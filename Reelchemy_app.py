#--------------------------------IMPORTS-------------------------------------------------------#
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import os
from dotenv import load_dotenv
import re


load_dotenv()

try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

#--------------------------------------DATA LOADING-------------------------------------------------------#

@st.cache_resource
def load_data():

    ratings = pd.read_parquet(
        "data/parquet/ratings.parquet"
    )

    movies = pd.read_parquet(
        "data/parquet/movies.parquet"
    )

    gscores = pd.read_parquet(
        "data/parquet/genome-scores.parquet"
    )

    links = pd.read_parquet(
        "data/parquet/links.parquet"
    )

    return ratings, movies, gscores, links


ratings, movies, gscores, links = load_data()

#--------------------------------------DATA PREPARATION-------------------------------------------------------#

@st.cache_data
def prep_data(ratings, movies, gscores, links):
    movies = pd.merge(movies, links, on = "movieId")
    movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    genre_dummies = movies["genres"].str.get_dummies(sep="|")
    genome_scores = gscores.pivot(
        index = "movieId",
        columns = "tagId",
        values = "relevance"
        )
    genome_scores_matrix = genome_scores.loc[genome_scores.index.isin(movies["movieId"])]
    C = ratings["rating"].mean()
    return movie_stats, genre_dummies, genome_scores, genome_scores_matrix, C, movies

movie_stats, genre_dummies, genome_scores, genome_scores_matrix, C, movies = prep_data(ratings, movies, gscores, links)

movies["title"] = movies["title"].str.replace(
    r"^(.*), The (\(\d{4}\))$",
    r"The \1 \2",
    regex=True
)

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

#--------------------------------------POSTER FUNCTIONS-------------------------------------------------------#

@st.cache_data
def get_poster(movieId):
    movie = movies.loc[movies["movieId"] == movieId]
    curr_id = movie["tmdbId"].iloc[0]
    if pd.notna(curr_id):
        curr_id = int(curr_id)
        curr_url = f"https://api.themoviedb.org/3/movie/{curr_id}"
        

        try:
            curr_response = requests.get(curr_url, params = {"api_key": TMDB_API_KEY}, timeout = 10)
            curr_response.raise_for_status()
            curr_data = curr_response.json()
            curr_path = curr_data.get("poster_path")
        except requests.exceptions.RequestException:
            return None
        except requests.exceptions.JSONDecodeError:
            return None
        
        if not curr_path:
            return None

        poster_url = f"https://image.tmdb.org/t/p/w500{curr_path}"
        return poster_url
    else:
        return None

def display_poster(poster_url, width):
    if poster_url:
        st.image(poster_url, width=width)
    else:
         st.image("Assets/poster_not_found_image.png", width=width)
    

#--------------------------------------Recommendation System-------------------------------------------------------#
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

        titled_hybrid_scores = pd.merge(
            hybrid_scores,
            movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        titled_hybrid_scores = pd.merge(
            titled_hybrid_scores,
            movie_stats,
            on="movieId",
            how="left"
        )
        return titled_hybrid_scores


#--------------------------------------STREAMLIT-------------------------------------------------------#
st.title("Reelchemy")
st.caption("Discover movies you'll love based on movies you already like!")

def reset_selection():
    st.session_state.selected_movie_id = None

movie_title = st.text_input("Search for a movie", on_change=reset_selection)

if movie_title and st.session_state.selected_movie_id is None:
    results = movies[movies["title"].str.contains(movie_title, case=False, na=False)][["movieId","title", "genres","tmdbId"]]

    if not results.empty:
        cols = st.columns(3)
        
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    movie_id = row["movieId"]
                    poster = get_poster(movie_id)
                    display_poster(poster, width=150)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")
                    if st.button("Select", key=f"select_{movie_id}"):
                        st.session_state.selected_movie_id = movie_id
                        st.rerun()
    else:
        st.write("no movies found")

if st.session_state.selected_movie_id is not None:
    st.button("Reset Selection", on_click=reset_selection)

    selected_movie = movies[movies["movieId"] == st.session_state.selected_movie_id].iloc[0]
    selected_poster = get_poster(st.session_state.selected_movie_id)
    st.subheader("Your Selected Movie")

    display_poster(selected_poster, width=200)
    st.write(selected_movie["title"])

    recommendations = recommend(st.session_state.selected_movie_id).head(21)
    st.subheader("Recommended Movies")
    rec_cols = st.columns(3)    

    for i, (_,row) in enumerate(recommendations.iterrows()):
        with rec_cols[i % 3]:
            with st.container(border=True):
                movie_id = row["movieId"]
                poster = get_poster(movie_id)
                display_poster(poster, width=150)
                st.markdown(f"**{row['title']}**")
                st.caption(f"Genres: {row['genres']}") 
                release_year = re.search(r"\((\d{4})\)$", row["title"])

                if release_year:
                    release_year = release_year.group(1)
                    st.caption(f"Released: {release_year}")
                st.caption(f"Average Rating: {row['mean']:.1f}/5")
                st.caption(f"Number of Ratings: {int(row['count']):,}")
    
st.divider()
st.caption("This app was created by Eric Winget using Streamlit and the MovieLens dataset. The movie posters are fetched from The Movie Database (TMDb) API.")
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")