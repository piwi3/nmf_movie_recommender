from app import app
from flask import render_template, request
from app.my_functions import nmf_recommender, svd_recommender, get_movie_list

@app.route("/")
@app.route("/home")
def homepage():
    movie_list = get_movie_list()
    return render_template("homepage.html", movie_list=movie_list)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/recommendations", methods=['GET', 'POST'])
def results():
    user_query = request.form.to_dict()
    user_query = {key:value for key, value in user_query.items() if value != ''}
  
    if user_query == {}:
        user_query = None

    #print(user_query) # for testing purpose
    
    if user_query != None:
        # Return result page if at least one movie has been rated
        top5_list = nmf_recommender(user_query)
        return render_template("results.html", top5_list=top5_list) 
    else:
        #Redirect to Homepage if no ratings have been submitted
        movie_list = get_movie_list()
        return render_template("homepage.html", movie_list=movie_list)