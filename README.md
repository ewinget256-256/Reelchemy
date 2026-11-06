# Reelchemy

This application allows you to search for a movie you like and gives recommendations based on a genre and genome tag based hybrid recommendation algorithm

## Overview

Reelchemy was created with the purpose of helping users find the next movie they will love. Instead of surfing streaming platforms for hours in search of the perfect movie, users simply input a movie they like and Reelchemy finds their next watch. Unlike a basic genre-matching recommender, Reelchemy combines genre similarity with MovieLens genome tag data to find movies with similar characteristics and provide more meaningful recommendations

## Features

- Movie Search and Selection -- Allows users to search for movies and select a recommendation starting point
- Movie Posters Displayed -- Retrieves posters using TMDb API
- Hybrid Recommendation System -- Combines genre and genome-tag similarity
- Individual Movie Data Insights -- Displays average movie rating and total rating counts

## Technologies Used

- **Python** -- Coding language used
- **Pandas** -- Dataset manipulation
- **NumPy** -- Used for numerical operations and calculations, including handling similarity scores
- **Scikit-learn** -- Used to calculate cosine similarity between movie feature vectors
- **Streamlit** -- Frontend interface
- **Requests** -- Used to send HTTP requests to the TMDb API
- **Python-Dotenv** -- Used to load TMDb API key from .env file
- **MovieLens Dataset** -- Source of movie and rating data
- **TMDb API** -- API used to obtain each movies respective poster

## Dataset

Reelchemy uses the MovieLens dataset, which contains each of the following files used within the application

- **movies.csv** -- provides movie titles and genres
- **ratings.csv** -- provides user ratings used to calculate average ratings and rating counts
- **genome-scores.csv** -- provides movie tag relevance scores used in genome similarity calculations
- **links.csv** -- provides IDs for external movie databases used to connect movies with TMDb and retrieve posters

## Recommendation Methods

**Genre Similarity** - Each movie's genre vector was converted into a binary vector consisting of 1s and 0s. A value of 1 indicates that a movie belongs to a particular genre, while 0 indicates that it does not. Cosine similarity was then used to compare movie genre vectors.

**Genome Similarity** - Each genome tag is assigned a relevance score for every movie within the dataset. I used these relevance scores to create a genome vector for each movie and applied cosine similarity to compare a selected movie against the rest of the dataset to find the best matches.

**Hybrid Recommendation System** - I used a weighted system giving genome similarity a heavier weight than genre similarity in order to obtain the best recommendations, additionally the system falls back to just genre similarity when it encounters a movie not included in the genome similarity dataset

## Weighted Rating

Throughout my project, my recommendation system uses a weighted rating formula that takes into account the amount of user ratings that come with the rating a movie is given. This is important because a movie could have a high average rating based on a few users. However a high rating backed by thousands of users is much more credible. The weighted rating formula helps account for this difference when ranking the movies for recommendation.

## How to use Reelchemy

1. Start by opening the Reelchemy website.
2. Search for a movie you would like recommendations based on.
3. Once your movie appears, click the **Select** button and wait for the recommendations to load.
4. Reelchemy will display your selected movie along with a list of recommended movies underneath.
5. To choose another movie, click **Select Another** or simply search for a different movie.

## Evaluation and Testing

To test my recommend function, I tested approximately 20 different movies across different popularity levels and genres. While testing, I found that the movies returned were great recommendations, but I did run into an unexpected error.

When finding the release year for movies, my function would automatically pull the last 4 characters of a movie's title, ignoring parentheses, to get the year. However, some movie titles didn't include the year. Because of this, I had to go back and create additional handling for gathering release years.

## Future improvements

- Ability to filter searches for things like genre, release year, or minimum rating
- Improving the user interface
- Allowing the user to pick multiple movies as a point of reference for recommendation