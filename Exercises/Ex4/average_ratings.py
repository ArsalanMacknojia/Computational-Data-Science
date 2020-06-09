# CMPT 353 - Exercise 4
# Arsalan Macknojia


import sys
import difflib
import pandas as pd


def avg_movie_ratings(movie_title, user_ratings):
    """
    Function compares titles present in user_ratings DF with movie_title and calculate the average rating.
    Rating is only accepted if the title in user_ratings matches with movie_title with at least 60% accuracy.
    """
    total_ratings = user_ratings.shape[0]
    matching_titles = difflib.get_close_matches(movie_title, user_ratings["title"], n=total_ratings, cutoff=0.6)

    filter = user_ratings['title'].isin(matching_titles)
    acceptable_ratings = user_ratings.loc[filter]

    return acceptable_ratings['rating'].mean()


def main():
    # Get file names
    movie_list = sys.argv[1]
    movie_ratings = sys.argv[2]
    output = sys.argv[3]

    # Store movie titles in a DF
    movie_titles = open(movie_list).readlines()
    movie_titles_df = pd.Series(movie_titles).replace('\n', '', regex=True)

    # Get user ratings data from .csv and calculate average mean rating for each movie title.
    user_ratings = pd.read_csv(movie_ratings)
    avg_ratings = movie_titles_df.apply(avg_movie_ratings, args=(user_ratings,))
    avg_ratings = round(avg_ratings, 2)

    # Add ratings column to DF with average movie rating for each title.
    result = pd.DataFrame(data={'title': movie_titles_df, 'rating': avg_ratings}).dropna().sort_values(by=['title'])
    result.to_csv(output)


if __name__ == '__main__':
    main()
