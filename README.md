## Deploying a movie recommender app with Flask & Heroku using Non-negative Matrix Factorization (NMF)
- Used __small data set__ of [MovieLense](https://grouplens.org/datasets/movielens/) and __webscraped movie posters__ from [Movieposterdb](https://www.movieposterdb.com)
- Implemented __NMF recommendation__ algorithm using the [surprise package](http://surpriselib.com) (surprise's NMF handles NaNs, Scikit's NMF does not)
- Built a __movie recommender app__ with __flask__ and __bootstrap__ and [deployed it with Heroku](https://hey-dude-what-to-watch-next.herokuapp.com) - 40 movies are selected (randomly from 10 clusters) and displayed; when user provides ratings, 5 movies are recommended based on predicted rating (using NMF)

<img src="https://github.com/piwi3/nmf_movie_recommender/blob/main/images/scrsht_movie_recommender.png" width="600"><br/>
_Figure 1: Frontend of movie recommender app (main page) built with flask and bootstramp_
