from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render,get_object_or_404,redirect
from django.db.models import Q
from django.http import Http404
from .models import Movie,Myrating
from django.contrib import messages
from .forms import UserForm
from django.db.models import Case, When
from .recommendation import Myrecommend
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def extract_genre_features():
    # Initialize an empty dictionary to store genre features
    genre_features = {}
    
    # Fetch all movies and their genres
    movies = Movie.objects.all()
    
    # For each movie, split the genre string into a list of genres
    # and store it in the dictionary with the movie's ID as the key
    for movie in movies:
        genres = movie.genre.split(',') # Assuming genres are comma-separated
        genre_features[movie.id] = genres
    
    return genre_features

# for recommendation
def recommend(request):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404
    
    # Extract genre features
    genre_features = extract_genre_features()
    
    # Collaborative filtering
    df = pd.DataFrame(list(Myrating.objects.all().values()))
    nu = df.user_id.unique().shape[0]
    current_user_id = request.user.id
    prediction_matrix, Ymean = Myrecommend()
    my_predictions = prediction_matrix[:, current_user_id - 1] + Ymean.flatten()
    pred_idxs_sorted = np.argsort(my_predictions)
    pred_idxs_sorted[:] = pred_idxs_sorted[::-1]
    pred_idxs_sorted = pred_idxs_sorted + 1
    
    # Exclude movies with the specified IDs
    excluded_movie_ids = [18, 11, 93, 55, 63]
    collaborative_recommendations = list(Movie.objects.filter(id__in=pred_idxs_sorted).exclude(id__in=excluded_movie_ids)[:4])
    
    # Content-based filtering
    # Convert genre features to a list of strings
    genre_list = [' '.join(genres) for genres in genre_features.values()]
    
    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer()
    genre_matrix = vectorizer.fit_transform(genre_list)
    
    # Calculate cosine similarity between movies
    cosine_sim = linear_kernel(genre_matrix, genre_matrix)
    
    # Get indices of movies in the dataset
    indices = pd.Series(movie.id for movie in Movie.objects.all()).reset_index(drop=True)
    
    # Get the top 4 most similar movies for each of the top 4 collaborative recommendations
    content_based_recommendations = []
    for movie in collaborative_recommendations:
        idx = indices[indices == movie.id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6] # Get the scores of the 5 most similar movies
        movie_indices = [i[0] for i in sim_scores]
        content_based_recommendations.extend(Movie.objects.filter(id__in=movie_indices).exclude(id__in=excluded_movie_ids))
    
    # Combine collaborative and content-based recommendations
    final_recommendations = collaborative_recommendations + content_based_recommendations[:4]
    
    return render(request, 'web/recommend.html', {'movie_list': final_recommendations})


# List view
def index(request):
    movies = Movie.objects.all()
    query = request.GET.get('q')
    excluded_movie_ids = [18, 11, 93, 55, 63]
    if query:
        movies = Movie.objects.filter(Q(title__icontains=query)).exclude(id__in=excluded_movie_ids).distinct()
    else:
        movies = Movie.objects.exclude(id__in=excluded_movie_ids)
    return render(request,'web/list.html',{'movies':movies})


# detail view
def detail(request,movie_id):
	if not request.user.is_authenticated:
		return redirect("login")
	if not request.user.is_active:
		raise Http404
	movies = get_object_or_404(Movie,id=movie_id)
	#for rating
	if request.method == "POST":
		rate = request.POST['rating']
		ratingObject = Myrating()
		ratingObject.user   = request.user
		ratingObject.movie  = movies
		ratingObject.rating = rate
		ratingObject.save()
		messages.success(request,"Your Rating is submited ")
		return redirect("index")
	return render(request,'web/detail.html',{'movies':movies})


# Register user
def signUp(request):
	form =UserForm(request.POST or None)
	if form.is_valid():
		user      = form.save(commit=False)
		username  =	form.cleaned_data['username']
		password  = form.cleaned_data['password']
		user.set_password(password)
		user.save()
		user = authenticate(username=username,password=password)
		if user is not None:
			if user.is_active:
				login(request,user)
				return redirect("index")
	context ={
		'form':form
	}
	return render(request,'web/signUp.html',context)				


# Login User
def Login(request):
	if request.method=="POST":
		username = request.POST['username']
		password = request.POST['password']
		user     = authenticate(username=username,password=password)
		if user is not None:
			if user.is_active:
				login(request,user)
				return redirect("index")
			else:
				return render(request,'web/login.html',{'error_message':'Your account disable'})
		else:
			return render(request,'web/login.html',{'error_message': 'Invalid Login'})
	return render(request,'web/login.html')

#Logout user
def Logout(request):
	logout(request)
	return redirect("login")




