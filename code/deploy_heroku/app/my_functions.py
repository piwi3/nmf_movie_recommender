from matplotlib.pyplot import get
import pandas as pd
import numpy as np
import random
from surprise import NMF, Dataset, Reader, SVD

base_path = 'app/data/'

def get_poster_path(movie_titles):
    """
    Get path of movie posters based on list of movie title.
    """
    #movie_posters = []
    #for title in movie_titles:
     #   filename = title.lower().replace('/', '_').replace(',','').replace(' ', '_').replace('(','').replace(')','').replace(':', '')
      #  filename = 'images/' + filename + '.jpg'
       # movie_posters.append(filename)

    df = pd.read_csv(f'{base_path}movie_poster_links.csv', index_col=0)
    movie_posters = []
    for title in movie_titles:
        filename = df.loc[title, 'img_link']
        movie_posters.append(filename)

    return movie_posters
    

def get_movie_list():
    """
    Selects movies to be evaluated by the visitor of the website.
    """
    # Load the data
    filename = base_path + 'movies_enriched.csv'
    df = pd.read_csv(filename, index_col=0)
    
    # Load 4 titles per cluster randomly with popularity (count of watched) as weights
    # And add movie poster information
    movie_list = []
    for cluster in range(10):
        movie_titles = df[df['cluster']==cluster].sample(n=4, weights='popularity')['title']
        movie_posters = get_poster_path(movie_titles)
        movie_cluster_list = zip(movie_titles, movie_posters)
        movie_list.append(movie_cluster_list)
    
    return movie_list

def create_dicts(df):
    """
    Creates dictionaries to enable conversion between movieIds and titles. Needs
    a dataframe with movieId and title columns as input (should be exhaustive).
    """
    # Create dictionary movieId -> title
    movieId_dict = {}
    for idx, movieId, title in df[['movieId', 'title']].itertuples():
        movieId_dict[movieId] = title

    # Create dictionary title -> movieId
    title_dict = {value:key for key, value in movieId_dict.items()}
        
    return movieId_dict, title_dict

def nmf_recommender(movie_ratings=None):
    """
    Recommender based on negative matrix factorization 
    (utilizing the surprise package)
    """
    # load data
    filename = base_path + 'full_table.csv'
    df = pd.read_csv(filename, index_col=0)
    
    # load dictionaries
    movieId_dict, title_dict = create_dicts(df)

    # Convert movie ratings into movie IDs
    movie_ratings = {
        title_dict[title]:rating for title, rating in movie_ratings.items()
        }

    # Create dataframe for new user
    df_new_usr = pd.DataFrame.from_dict(movie_ratings, orient='index')
    df_new_usr['userId']='new_user'
    df_new_usr = df_new_usr.reset_index().rename(columns={'index':'movieId', 0:'rating'})
    df_new_usr = df_new_usr[['userId', 'movieId', 'rating']]

    # Concat dataframes (i.e. create one single / new dataset)
    df_nmf = df[['userId', 'movieId', 'rating']]
    df_nmf = pd.concat([df_nmf, df_new_usr], ignore_index=True)
    
    # Load the data
    reader = Reader(rating_scale=(0, 5)) # Define the scale of the ratings
    data = Dataset.load_from_df(df_nmf, reader)
    
    # Define the whole dataset as trainset
    trainset = data.build_full_trainset() 

    # Create the model, n_features and n_epochs optimized via grid search
    n_factors = 5 
    n_epochs = 200
    model = NMF(n_factors=n_factors, n_epochs=n_epochs) 

    # Fit the model
    model.fit(trainset)

    # Generate test_data (i.e. movies not yet watched by new_user)
    data_all_users = trainset.build_anti_testset()
    test_data = [set for set in data_all_users if set[0] == 'new_user']
        
    # Get predictions on test dataset
    test_results = model.test(test_data)

    # Save results in a dataframe
    df_test_results = pd.DataFrame(np.array(test_results)[:,(1,3)], columns=['movieId', 'prediction'])
    df_test_results['userId'] = 'new_user'
    df_test_results = df_test_results[['userId', 'movieId', 'prediction']]
    df_test_results

    # Generate final result
    df_test_results['title'] = df_test_results['movieId'].apply(lambda x: movieId_dict[x])
    top5 = df_test_results.sort_values('prediction', ascending=False)['title'].head(5)

    # Add movie poster info
    movie_posters = get_poster_path(top5)
    top5_list = zip(top5, movie_posters)

    return top5_list


def svd_recommender(movie_ratings=None):
    """
    Recommender based on single value decomposition (SVD) popularized by Simon Funk during the 
    Netflix Prize (utilizing the surprise package)
    """
    # load data
    filename = base_path + 'full_table.csv'
    df = pd.read_csv(filename, index_col=0)
    
    # load dictionaries
    movieId_dict, title_dict = create_dicts(df)

    # Convert movie ratings into movie IDs
    movie_ratings = {title_dict[title]:rating for title, rating in movie_ratings.items()}

    # Create dataframe for new user
    df_new_usr = pd.DataFrame.from_dict(movie_ratings, orient='index')
    df_new_usr['userId'] = 'new_user'
    df_new_usr = df_new_usr.reset_index().rename(columns={'index':'movieId', 0:'rating'})
    df_new_usr = df_new_usr[['userId', 'movieId', 'rating']]

    # Concat dataframes (i.e. create one single / new dataset)
    df_nmf = df[['userId', 'movieId', 'rating']]
    df_nmf = pd.concat([df_nmf, df_new_usr], ignore_index=True)
    
    # Load the data
    reader = Reader(rating_scale=(0, 5)) # Define the scale of the ratings
    data = Dataset.load_from_df(df_nmf, reader)
    
    # Define the whole dataset as trainset
    trainset = data.build_full_trainset() 

    # Create the model, n_features and n_epochs optimized via grid search
    n_factors = 5 
    n_epochs = 20
    model = SVD(n_factors=n_factors, n_epochs=n_epochs) 

    # Fit the model
    model.fit(trainset)

    # Generate test_data (i.e. movies not yet watched by new_user)
    data_all_users = trainset.build_anti_testset()
    test_data = [item for item in data_all_users if item[0] == 'new_user']
    
    # Get predictions on test dataset
    test_results = model.test(test_data)
    
    # Save results in a dataframe
    df_test_results = pd.DataFrame(np.array(test_results)[:,(1,3)], columns=['movieId', 'prediction'])
    df_test_results['userId'] = 'new_user'
    df_test_results = df_test_results[['userId', 'movieId', 'prediction']]
    df_test_results

    # Generate final result
    df_test_results['title'] = df_test_results['movieId'].apply(lambda x: movieId_dict[x])
    top5 = df_test_results.sort_values('prediction', ascending=False)['title'].head(5)

    # Add movie poster info
    movie_posters = get_poster_path(top5)
    top5_list = zip(top5, movie_posters)

    return top5_list

import plotly.graph_objects as go


if __name__ == '__main__':
    """
    This enables test running the functions / code
    """
    # test nmf recommender
    movie_ratings = {
    'Toy Story (1995)' : 2,
    'Batman Begins (2005)' : 5,
    'Inception (2010)' : 4,
    'Heat (1995)' : 3,
    }
    top5_list = nmf_recommender(movie_ratings)
    print(top5_list)
    
    
    