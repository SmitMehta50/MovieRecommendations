from flask import Flask, render_template, request
import flask
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import nltk
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display,YouTubeVideo
from youtubesearchpython import VideosSearch
from skimage import io
import json, requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

api_key='ae425b92085fb0baed654d771acaed36'

# In[ ]:


def get_url(url,path):
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text)
    if python_dictionary_values['results']:
        profile_path=python_dictionary_values['results'][0][path]
        if profile_path == None :
            return None,python_dictionary_values['results'][0]
        else:
            return profile_path,python_dictionary_values['results'][0]


# In[ ]:


# def print_image(mylist,mysize,mynames):
#     plt.figure(figsize=mysize)
#     for i in range(len(mylist)):
#         url=mylist[i]
#         image = io.imread(url)
#         plt.subplot(3,4, i+1)
#         plt.imshow(image)
#         plt.title(mynames[i])
#     plt.tight_layout()
#     plt.show()


# In[ ]:


def get_img_url(url,path):
    profile_path,x=get_url(url,path)
    if profile_path == None:
        x='https://m.media-amazon.com/images/S/sash/N1QWYSqAfSJV62Y.png'
    else:
        profile_image='https://image.tmdb.org/t/p/w185/'+ profile_path
        x=profile_image
    return x


# In[ ]:


def get_dict(url):
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text)
    return python_dictionary_values


# In[ ]:


def find_movies_names(python_dictionary_values):
    x=python_dictionary_values['results'][0]
    profile_path = x['poster_path']
    profile_name = str(x['original_title']) + ' (' + str(x['release_date'][:4]) + ')'
    if profile_path == None  :
        profile_path='https://m.media-amazon.com/images/S/sash/N1QWYSqAfSJV62Y.png'
    else:
        profile_path='https://image.tmdb.org/t/p/w185/'+ profile_path
        
    return profile_path,profile_name


# In[ ]:


def search_person(mylist,mynames,search):
    path='profile_path'
    mylst=[]
    for i in mylist:
        s=i.replace(' ','%20')
        url = 'https://api.themoviedb.org/3/search/'+search+'?api_key='+api_key+'&language=en-US&query=%27'+s+'%27&page=1&include_adult=false'
        mylst.append(get_img_url(url,path))
    # for i in range(len(mylst)):
        # print(mynames[i])
        # search_person_name = mynames[i]
        # print(mylst[i])
        # search_person_lst = mylst[i]
    #print_image(mylst,mysize,mynames)
    return mynames, mylst


# In[ ]:


def search_movie(mylist,mynames,search):
    mylst=[]
    mynames=[]
    for i in mylist:
        s=i.replace(' ','%20')
        url = 'https://api.themoviedb.org/3/search/'+search+'?api_key='+api_key+'&language=en-US&query=%27'+s+'%27&page=1&include_adult=false'
        x,y=find_movies_names(get_dict(url))
        mylst.append(x)
        mynames.append(y)
    # for i in range(len(mylst)):
        # print(mynames[i])
        # print(mylst[i])
    #print_image(mylst,mysize,mynames)
    return mynames, mylst


# In[ ]:


def get_detail(d,s):
    lst=[]
    for i in range(len(d[s])):
        lst.append(d[s][i]['name'])
    return lst


# In[ ]:


def get_genre(idx):
    url = 'https://api.themoviedb.org/3/movie/'+str(idx)+'?api_key='+api_key+'&language=en-US'
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text)
    lst = get_detail(python_dictionary_values,'genres')
    return lst,python_dictionary_values


# In[ ]:


def movie_details(movie):
    s=movie.replace(' ','%20')
    url='https://api.themoviedb.org/3/search/'+'movie'+'?api_key='+api_key+'&language=en-US&query=%27'+s+'%27&page=1&include_adult=false'
    path='id'
    idx,mymovie=get_url(url,path)
    my_genre,mymovie=get_genre(idx)
    
    title=mymovie['original_title']
    # print('Movie name : \'{}\' '.format(title))
    
    profile_image=[]
    if mymovie['poster_path'] != None:
        profile_image.append('https://image.tmdb.org/t/p/w185/'+ mymovie['poster_path'])
    else :
        profile_image.append('https://m.media-amazon.com/images/S/sash/N1QWYSqAfSJV62Y.png')
        
    if mymovie['backdrop_path'] != None:
        profile_image.append('https://image.tmdb.org/t/p/w185/'+ mymovie['backdrop_path'])
        
    # print(profile_image[0])
    movie_image1 = profile_image[0]
    # print(profile_image[1])
    movie_image2 = profile_image[1]
        
    #print_image(profile_image,mysize=(15,10),mynames=[' ',' '])
    
    imdb_id = mymovie['imdb_id']
    
    # print('\nMovie plot :')
    # print(mymovie['overview'])
    mymovie['overview']
    
    # print('\nMovie Genre :')
    # print(my_genre)
    movie_genre = my_genre
    
    # print('\nOriginal_language : {}'.format(mymovie['original_language']))
    mymovie['original_language']
    
    # print('\nMovie released date : {}'.format(mymovie['release_date']))
    date=str(mymovie['release_date'][:4])
    mymovie['release_date']
    
    # print('\nMovie ratings : {}({})'.format(mymovie['vote_average'],mymovie['vote_count']))
    mymovie['vote_average']
    mymovie['vote_count']
    
    # print('\nproduction_companies : {}'.format(get_detail(mymovie,'production_companies')))
    movie_production_companies = get_detail(mymovie,'production_companies')
    
    # print('\nMovie length : {} minutes '.format(str(mymovie['runtime'])))
    str(mymovie['runtime'])

    # print('\nMovie tagline: \'{}\' '.format(str(mymovie['tagline'])))
    str(mymovie['tagline'])
    
    url='https://api.themoviedb.org/3/movie/'+str(idx)+'/credits?api_key='+api_key+'&language=en-US'
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text) 
    
    df=pd.DataFrame(python_dictionary_values['cast'])
    df['mynames'] = df['original_name'] + ' as ' + df['character']
    df2=pd.DataFrame(python_dictionary_values['crew'])
    
    # print('\nTop cast :')
    myactors = df[df['known_for_department']=='Acting'].original_name.tolist()
    mynames = df[df['known_for_department']=='Acting'].mynames.tolist()
    movie_cast, cast_url = search_person(myactors[:10],mynames[:10],search='person')
    x=myactors[:12]        
    
    # print('\nDirectors of the movie :')
    myactors = df2[df2['department']=='Directing'].original_name.tolist()
    movie_directors, directors_url = search_person(myactors[:4],myactors[:4],search='person')
    y=myactors[:4]
        
    # print('\nWriters of the movie :')
    myactors = df2[df2['department']=='Writing'].original_name.tolist()
    movie_writers, writers_url = search_person(myactors[:4],myactors[:4],search='person')
    z=myactors[:4]
         
    return title, date,imdb_id,x,y,z,idx, mymovie, movie_image1, movie_image2, movie_genre, movie_production_companies, movie_cast, movie_directors, movie_writers, cast_url, directors_url, writers_url


# In[ ]:


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def remove_special_characters(text):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,' ',str(text))
    text=re.sub(r'\s+',' ',text)
    return text


# In[ ]:


stop = stopwords.words('english')
stoplist=[]
for i in stop:
    if '\'' in i:
        stoplist.append(i.replace('\'', ''))
    else:
        stoplist.append(i)
stop = stoplist


# In[ ]:


lem=WordNetLemmatizer()
def Process(sent):
    result=''
    sent=sent_tokenize(sent)
    for i in range(len(sent)):
        sent[i]=sent[i].lower()
        words=word_tokenize(sent[i])
        new=[]
        for word in words:
            if word not in stop:
                new.append(word)
        new=[lem.lemmatize(new_word) for new_word in new]
        sent[i]=' '.join(new)  
        result+=sent[i]
    return result


# In[ ]:


def imdb(idx,myactors,mydirectors,mywriters):
    myactors = [x.lower() for x in myactors]
    mydirectors = [x.lower() for x in mydirectors]
    mywriters = [x.lower() for x in mywriters]
    
    myactors = [x.replace(' ','') for x in myactors]
    mydirectors = [x.replace(' ','') for x in mydirectors]
    mywriters = [x.replace(' ','') for x in mywriters]
    
    my_list = list(set(myactors + mydirectors + mywriters))
    my_list=' '.join(my_list)
    # print(my_list)
    
    url='https://api.themoviedb.org/3/movie/'+str(idx)+'?api_key='+api_key+'&language=en-US'
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text)
    
    genres=[]
    for i in range(len(python_dictionary_values['genres'])):
        genres.append(python_dictionary_values['genres'][i]['name'])
    genres=' '.join(genres)
    genres=genres.replace('Science Fiction','SciFi')
    genres=genres.replace('TV Movie','RealityTV')
    
    title=python_dictionary_values['original_title'].lower()
    
    des = python_dictionary_values['overview']
    des = remove_special_characters(des)
    des = Process(des)
    
    val=[[title,genres,des,my_list]]
    col=['original_title','genre','description','full_cast']
    d=pd.DataFrame(data=val,columns=col)
    
    # print(d)
    
    return d


# In[ ]:


def trending():
    url='https://api.themoviedb.org/3/movie/popular?api_key='+api_key+'&language=en-US&page=1'
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text)
    
    myname=[]
    myposter=[]
    for i in range(len(python_dictionary_values['results'])):
        x=python_dictionary_values['results'][i]
        s=x['original_title']+' ('+x['release_date'][:4]+')'
        myname.append(s)
        s='https://image.tmdb.org/t/p/w185/'+x['poster_path']
        myposter.append(s)
# If does not work check here for loop
    # for i in range(10):
    #     print(myname[i])
    #     print(myposter[i])

    return myname, myposter
    #print_image(mylist=myposter[:10],mynames=myname[:10],mysize=(15,8))


# In[ ]:


def metascore(mylist,df,vect):
    name=[]
    vectorizer = CountVectorizer(lowercase = False)
    for i in mylist:
        if i=='genre':
            vectorizer = CountVectorizer(lowercase = False,ngram_range=(1,3))
        data_vect = vectorizer.fit_transform(df[i].fillna('None'))
        vect1 = vectorizer.transform(vect[i])
        sim_score = cosine_similarity(data_vect,vect1)
        name.append('sim_score_'+i)
        df['sim_score_'+i] = sim_score
    return name,df


# In[ ]:


def string_match(name,th=0.9):
    vectorizer = CountVectorizer(lowercase = False)
    for i in range(len(name)):
        l=0
        if (i == 11) | (i >= len(name)):
            break
        vec1=vectorizer.fit_transform([name[i]])
        for j in range(i+1,len(name)):
            vec2=vectorizer.transform([name[j-l]])
            sim_score = cosine_similarity(vec1,vec2)
            if sim_score[0][0] >= th:
                index=j-l
                name = name[:index] + name[index+1 :]
                l+=1
    return name[:10]





def cast_movie(name):
    df=pd.read_csv(r'myfinaldata4.csv')
    search='person'
    s=name.replace(' ','%20')
    url = 'https://api.themoviedb.org/3/search/'+search+'?api_key='+api_key+'&language=en-US&query=%27'+s+'%27&page=1&include_adult=false'
    response = requests.get(url)
    python_dictionary_values = json.loads(response.text)
    actor_name=python_dictionary_values['results'][0]['name']
    profile_path=python_dictionary_values['results'][0]['profile_path']
    profile_image = ''
    if profile_path == None:
        x='https://m.media-amazon.com/images/S/sash/N1QWYSqAfSJV62Y.png'
    else:
        profile_image='https://image.tmdb.org/t/p/w185/'+ profile_path
    # print(profile_image)
    #print_image([profile_image],mynames=[actor_name],mysize=(15,10))
    
    d=python_dictionary_values['results'][0]['known_for_department']
    # print('known_for_department :{}\n'.format(d))
    myname=actor_name
    actor_name = actor_name.lower().replace(' ','')
    df=df[df.full_cast.fillna('None').str.contains(actor_name)]
    sorted_data = df.sort_values(by=['metascore1'],ascending=False)
    name = sorted_data.iloc[:,[1]].values.ravel().tolist() 
    name= string_match(name,th=0.9)
    
    # print('Top movies by {}'.format(myname))
    search_movie(name,name,search='movie')
    
    return 0,0,0


# In[ ]:


def similarity2(movie,mylist=['genre','full_cast'],weight='balanced',pop=5,start_year=2000,end_year=2020,min_rating=5.0,Total_votes=100000):
    df=pd.read_csv(r'myfinaldata4.csv')
    my_movie,mymoviedate,myid,my_actor,my_director,my_writer,idx, mymovie, movie_image1, movie_image2, movie_genre, movie_production_companies, movie_cast, movie_directors, movie_writers, cast_url, directors_url, writers_url = movie_details(movie)
    # print('\nMovie Trailer :\n')
    trailer_search = my_movie + ' ' + mymoviedate + ' official' + ' trailer'
    videosSearch = VideosSearch(trailer_search, limit = 1)
    result=videosSearch.result()
    # print(result['result'][0]['link'])
    trailer = result['result'][0]['link']
    #idx=result['result'][0]['id']
    #display(YouTubeVideo(id=idx,width=900,height=500))
    vect = df.loc[df['imdb_title_id'] == str(myid)]
    if vect.empty:
        vect = imdb(idx,my_actor,my_director,my_writer)
    min_rating=(min_rating/10.0)
    Total_votes=(Total_votes/2278845.0)
    df=df[((df.year >= start_year) & (df.year <= end_year)) & (df.votes >= Total_votes) & (df.avg_vote >= min_rating)]
    if df.empty:
        print('Srry movies not in database.Check again parameters entered.No recommendation can be generated')
    else:
        #print(vect)
        # print('****************************************************************************************')
        # print('\nRecommended Movies :')
        name,df=metascore(mylist,df,vect)
        if (weight == 'balanced') | (len(mylist)==1):
            weights=[(1.0/len(mylist))]*len(mylist)
        else:
            s=sum(weight)
            weights = [i/s for i in weight]
        df['metascore2']=0.0
        for i in range(len(name)):
            df['metascore2']=df['metascore2'] + df[name[i]]*weights[i]
        my_weight=(pop/10.0)
        df['metascore'] = df['metascore1']*my_weight + df['metascore2']*(1.0-my_weight)
        sorted_data = df.sort_values(by=['metascore'],ascending=False)
        #print(sorted_data.loc[:,['original_title','votes','avg_vote','genre','metascore2','metascore1']].head(12))
        name = sorted_data.iloc[:21,[1]].values.ravel().tolist() 
        #name.remove(my_movie.lower())
        name= string_match(name,th=0.9)       
        recommend_movie, recommend_url = search_movie(name,name,search='movie')
        # print(recommend_movie)
        # print(recommend_url)
        
    # print('*******************************************************************************************************')
    # print('Top movies from the director:\n')
    x,y,z = cast_movie(my_director[0])
        
    # print('*******************************************************************************************************')
    # print('Top movies from the lead actor:\n')
    x,y,z = cast_movie(my_actor[0])
        
    if my_director[0] != my_writer[0]:
        # print('*******************************************************************************************************')
        # print('Top movies from the writer:\n')
        x,y,z = cast_movie(my_writer[0])
        
    # print('Trending today:\n')
    myname, myposter = trending()
        #return my_movie,movies,movies_names
    return my_movie, mymoviedate, my_actor, trailer, my_director, my_writer, mymovie, movie_image1, movie_image2, movie_genre, movie_production_companies, movie_cast, movie_directors, movie_writers, cast_url, directors_url, writers_url, recommend_movie, recommend_url






app = Flask(__name__)

@app.route('/home', methods=['GET','POST'])
@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        myname, myposter = trending()
        
        return(flask.render_template('index.html', myname=myname, myposter=myposter ))


@app.route('/recommendations', methods=['POST'])
def recommendations():            
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        start_date = 1990
        end_date = 2020
        popularity = 0
        rating = 5.0
        no_votes = 100000
        mylist = ['genre']
        myweight = [1.0]
        movie_name = m_name

        my_movie, mymoviedate, my_actor, trailer, my_director, my_writer, mymovie, movie_image1, movie_image2, movie_genre, movie_production_companies, movie_cast, movie_directors, movie_writers, cast_url, directors_url, writers_url, recommend_movie, recommend_url = similarity2(movie_name,mylist,myweight,popularity,start_date,end_date,rating,no_votes)
        

        return(flask.render_template('positive.html', title = my_movie,
        moviedate = mymoviedate,
        movieactor = movie_cast,
        actorurl= cast_url,
        youtube_link = trailer,
        director = movie_directors,
        directorurl = directors_url,
        writer = movie_writers,
        writerurl=writers_url,
        moviedata = mymovie,
        image1 = movie_image1,
        image2 = movie_image2,
        genre = movie_genre,
        companies = movie_production_companies,
        recommendmovie = recommend_movie,
        recommendurl = recommend_url ))


if __name__ == "__main__":
    app.run(debug = True, port=5000)