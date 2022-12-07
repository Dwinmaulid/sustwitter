from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, flash, request, session
import json
import os
import tweepy
from werkzeug.utils import secure_filename


consumer_key = os.getenv('API_KEY')
consumer_secret = os.getenv('API_SECRET_KEY')
callback = 'http://127.0.0.1:5000/callback'
bearer_token = os.getenv('BEARER_TOKEN')


app = Flask(__name__)
app.secret_key = os.urandom(12)  # Generic key for dev purposes only
ALLOWED_EXTENSIONS = set(['csv','xlsx'])
UPLOAD_FOLDER = './dataset/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))

# ======== Menu ============================================================== #
# -------- Home -------------------------------------------------------------- #
@app.route("/home")
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        user = helpers.get_user()
        return render_template('home.html', user=user)

# ======== Apps ============================================================== #
# -------- Crawling Dataset -------------------------------------------------- #
@app.route("/apps")
def apps():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
    return render_template('apps.html', auth=auth)

@app.route('/apps/auth')
def auth():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
    url = auth.get_authorization_url()
    session['request_token'] = auth.request_token
    return redirect(url)

@app.route('/apps/callback')
def twitter_callback():
    request_token = session['request_token']
    del session['request_token']

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
    auth.request_token = request_token
    verifier = request.args.get('oauth_verifier')
    auth.get_access_token(verifier)
    session['token'] = (auth.access_token, auth.access_token_secret)

    return redirect('/apps_menu')

@app.route('/apps_menu')
def request_twitter():
    token, token_secret = session['token']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
    auth.set_access_token(token, token_secret)
    api = tweepy.API(auth)

    return render_template('apps.html', api=api)

# -------- Crawling Data ----------------------------------------------------- #
@app.route('/apps_menu/crawling')
def crawling_data():
    token, token_secret = session['token']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
    auth.set_access_token(token, token_secret)
    api = tweepy.API(auth)
    
    search_tweets = "#covid -filter:retweets"
    tweets = tweepy.Cursor(api.search_tweets,q=search_tweets,since="2022-04-01", lang='en').items(500)

    return render_template('crawling.html', tweets=tweets)

@app.route("/apps_menu/crawling/download")
def download():
    return render_template('apps.html', files=os.listdir('dataset'))

# -------- Check Account ----------------------------------------------------- #
@app.route('/apps_menu/check-account', methods=['GET','POST'])
def check_account():
    token, token_secret = session['token']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback)
    auth.set_access_token(token, token_secret)
    api = tweepy.API(auth)

    if request.method == "POST":
        screen_name = request.form["username"]
        user = api.get_user(screen_name = request.form["username"])

        # feature tweet of user
        tweets_list= api.user_timeline(screen_name = request.form["username"], count=1) 
        tweet= tweets_list[0]

        profile_picture=user.profile_image_url
        profile_picture=profile_picture.replace('_normal', '')

        # info profile user
        display_name = user.name
        location = user.location 
        description = user.description
        website = user.url
        verified = user.verified
        followers = user.followers_count
        followings = user.friends_count
        public_member = user.listed_count
        likes = user.favourites_count 
        total_tweets = user.statuses_count
        created_at = user.created_at
        default_profile = user.default_profile
        default_profile_image = user.default_profile_image 

        # info tweet user
        tweet_source = tweet.source 
        tweet_geotagged = tweet.place
        tweet_retweets = tweet.retweet_count
        tweet_likes = tweet.favorite_count
        tweet_mentions = tweet.entities['user_mentions']
        tweet_hashtags = tweet.entities['hashtags']
        tweet_links = tweet.entities['urls']
        
        ##### Preprocessing khusus atribut lolos ig #####
        import pandas as pd
        import numpy as np
        from datetime import datetime, timezone
        from scipy.stats import zscore
        from sklearn import svm
        from sklearn.model_selection import KFold, cross_val_predict
        import time as time

        # atribut location

        location2 = location

        if location2 == "":
            without_location = 1  
        else:
            without_location = 0

        # atribut description
        description2 = description

        if description2 == '':
            without_description = 1
        else:
            without_description = 0

        # atribut website
        website2 = website

        if website2 == '':
            without_website = 1
        else:
            without_website = 0

        # atribut verified
        verified2 = verified

        if verified2 == True:
            unverified = 0
        else:
            unverified = 1


        # atribut default theme
        default_theme2 = default_profile

        if default_theme2 == True:
            default_theme = 1
        else:
            default_theme = 0

        # atribut age 
        today = pd.datetime.now(timezone.utc)

        created_at2 = pd.to_datetime(created_at, infer_datetime_format = True)
        date_created = abs(today-created_at2).days< 180

        if date_created == True:
            age = 1
        else:
            age = 0

        # atribut source
        source = len(tweet_source)

        # atribut mentions
        mentions = sum('screen_name' in s for s in tweet_mentions)

        # atribut hashtags
        hashtags = sum('text' in s for s in tweet_hashtags)

        ### buat dataframe dari akun yg dicari ###
        
        # buat list data indeks kolom dan isinya
        data = {
            "Without location" : [without_location], 
            "Without description" : [without_description],
            "Without website" : [without_website],
            "Unverified" : [unverified], 
            "Followers" : [followers],
            "Followings" : [followings],
            "Public member" : [public_member], 
            "Likes" : [likes],
            "Total tweets" : [total_tweets],
            "Age" : [age], 
            "Default theme" : [default_theme],
            "Source" : [source],
            "Tweet's retweets" : [tweet_retweets],
            "Tweet's likes" : [tweet_likes], 
            "Mentions" : [mentions],
            "Hashtags" : [hashtags]
            }

        # input data ke dataframe
        df = pd.DataFrame(data, index=[0])


        #### PROSES LDA ####
        # Logarithm Transformation
        df1=pd.read_csv('dataset\ig.csv')
        df2=pd.concat([df, df1], ignore_index=True, sort=False)
        df2['Label'] = df2['Label'].fillna(0)
        X2 = df2.iloc[:,:-1].values
        y2 = df2.iloc[:,-1]

        column=["Followers","Followings","Public member", "Likes","Total tweets"]
        df2[column]=df2[column].applymap(lambda x: np.log(x) if x > 0 else np.log(x+1))

        # Z-score
        df3=df2.apply(zscore)

        # LDA model
        class LDA:

            # Objek __init__  menerima konstanta sebagai argumen
            def __init__(self, n_components):
                self.n_components = n_components
                self.linear_discriminants = None

            def fit(self, X, y):
                n_features = X.shape[1]     #berikan jumlah baris dalam array
                class_labels = np.unique(y) #dapatkan nilai unik dari array yang diberikan sebagai parameter

                # Scatter matrix dalam kelas:
                # SW = sum((X_c - mean_X_c)^2 )

                # Scatter matrix antar kelas:
                # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

                mean_overall = np.mean(X, axis=0)
                SW = np.zeros((n_features, n_features))
                SB = np.zeros((n_features, n_features))
                for c in class_labels:
                    X_c = X[y == c]
                    mean_c = np.mean(X_c, axis=0)
                    # (20, n_c) * (n_c, 20) = (20,20) -> transpose
                    SW += (X_c - mean_c).T.dot((X_c - mean_c))

                    # (20, 1) * (1, 20) = (20,20) -> reshape
                    n_c = X_c.shape[0]
                    mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
                    SB += n_c * (mean_diff).dot(mean_diff.T)

                # Menghitung SW^-1 * SB
                A = np.linalg.inv(SW).dot(SB)
                # Mencari nilai eigen dan vektor eigen dari SW^-1 * SB
                eigenvalues, eigenvectors = np.linalg.eig(A)
                # -> vektor eigen v = [:,i] vektor kolom, transpose untuk mempermudah kalkulasi
                # urutan vektor eigen dari tinggi ke rendah
                eigenvectors = eigenvectors.T
                idxs = np.argsort(abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[idxs]
                eigenvectors = eigenvectors[idxs]
                # simpan vektor eigen dengan nilai eigen terbesar
                self.linear_discriminants = eigenvectors[0:self.n_components]

            def transform(self, X):
                # project data
                return np.dot(X, self.linear_discriminants.T)

        #Terapkan LDA ke model
        X3 = df3.iloc[:,:-1]
        y3 = df3.iloc[:,-1]
        y3 = y3.astype('int')

        #proyeksi data ke bentuk 1 lD utama
        lda = LDA(1)
        lda.fit(X3.values,y3.values)
        X3_projected = lda.transform(X3)

        #Hasil LDA
        X3_lda = X3_projected.real
        df3['LDA'] = X3_lda
        LDA_result = pd.DataFrame(df3['LDA'])

        df4 = pd.concat([LDA_result, y2], axis=1)
        df4 = df4.round(decimals=3)
        X4 = df4.iloc[:,:-1].values
        y4 = df4.iloc[:,-1].values
        n4 = len(df4)

        #### PROSES SVM ####
        # Perhitungan SVM+IG+LDA

        # baca data model
        # Pembagian data dengan k-fold cross validation dimana k=14
        num_folds = 14
        kfold = KFold(n_splits=num_folds)

        #Terapkan model
        clf = svm.SVC(kernel='rbf', C=1000, gamma=0.5) 
        model = cross_val_predict(clf, X4, y4.ravel(), cv=kfold)
        y_predict = model[0]
        
        if y_predict  == 0:
            prediction= 'Real Account'
        else:
            prediction = 'Fake Account'

        return render_template('check-account.html',
        profile_picture=profile_picture,
        username=screen_name,
        display_name=display_name,
        location=location,
        description=description,
        website=website,
        verified=verified,
        followers=followers,
        followings=followings,
        public_member=public_member,
        likes=likes,
        total_tweets=total_tweets,
        created_at=created_at,
        default_profile=default_profile,
        default_profile_image=default_profile_image,
        tweet_source = tweet_source,
        tweet_geotagged = tweet_geotagged,
        tweet_retweets = tweet_retweets, 
        tweet_likes = tweet_likes,
        tweet_mentions = tweet_mentions,
        tweet_hashtags = tweet_hashtags, 
        tweet_links = tweet_links,
        prediction=prediction,
        is_post=True
        )
  
    return render_template('check-account.html', api=api)


# -------- Demonstration ----------------------------------------------------- #
@app.route('/demonstration.html') 
def demonstration():
    return render_template('demonstration.html')


@app.route('/demonstration.html', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # periksa apakah post request memiliki bagian file
        if 'file' not in request.files:
            flash('Tidak ada file')
            return render_template('demonstration.html')
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang diunggah')
            return render_template('demonstration.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash('telah diunggah, silakan klik tampil data untuk menampilkan data')
            return render_template('demonstration.html', filename=filename)
        else:
            flash('Format file yang diunggah harus dalam .csv')
            return render_template('demonstration.html')

@app.route('/upload_demonstration_dataset')
def showData():
    import pandas as pd
    import numpy as np
    from scipy.stats import zscore
    from sklearn import svm
    from sklearn.model_selection import KFold, cross_val_score
    import time as time

    #### DATA AWAL (harus sudah melakukan preprocessing) ####
    # Menampilkan file upload dari session
    data_file_path = session.get('uploaded_data_file_path', None)
 
    # read csv file from lokasi server upload
    df = pd.read_csv(data_file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1]
    n = len(df)

    #### PROSES IG ####
    # nilai df, X, dan y untuk IG masih sama dgn upload_df

    # entropy formula
    def hitung_impurity(X, kriteria_impurity):
        
        probs = X.value_counts(normalize=True)
        
        if kriteria_impurity == 'entropi':
            impurity = -1 * np.sum(np.log2(probs) * probs)
        else:
            raise ValueError('Kriteria impurity tidak diketahui')
            
        return(round(impurity, 3))

    # Gain formula
    def hitung_information_gain_X(df, y, atribut, kriteria):
                
        entropi_y = hitung_impurity(df[y], kriteria)

        list_entropi = list()
        list_bobot = list()

        for value in df[atribut].unique():
            nilai_df_X = df[df[atribut] == value]
            nilai_entropi = hitung_impurity(nilai_df_X [y], kriteria)
            list_entropi.append(round(nilai_entropi, 3))
            nilai_bobot = len(nilai_df_X ) / len(df)
            list_bobot.append(round(nilai_bobot, 3))

        sisa_impurity_X = np.sum(np.array(list_entropi) * np.array(list_bobot))
        information_gain = entropi_y - sisa_impurity_X
        information_gain=np.around(information_gain, decimals=3)
        return(atribut, information_gain)

    IG = []

    kriteria = 'entropi'
    for X in df.drop(columns='Label').columns:
        info_gain_X = hitung_information_gain_X(df, 'Label', X, kriteria)
        IG.append(info_gain_X)
        
    # Data of Gain with threshold
    df2 = pd.DataFrame(IG, columns = ['Atribut','Gain'])
    threshold = 0.02

    # Retrive dataframe of Gain
    columns = []
    for index, row in df2.iterrows():
        if row['Gain'] > threshold:
            x = row['Atribut']
            columns.append(x)

    df3 = pd.concat([df[columns], y], axis=1)
    X3 = df3.iloc[:,:-1].values
    y3 = df3.iloc[:,-1].values
    n3 = len(df3)

    X2 = np.array(df2['Atribut']) 
    y2 = np.array(df2['Gain']) 
    n2 = len(df2)


    #### PROSES LDA ####
    # Logarithm Transformation
    df4 = df3.copy()
    column=["Followers","Followings","Public member", "Likes","Total tweets"]
    df4[column]=df4[column].applymap(lambda x: np.log(x) if x > 0 else np.log(x+1))

    # Z-score
    df5=df4.apply(zscore)

    # LDA model
    class LDA:

        # Objek __init__  menerima konstanta sebagai argumen
        def __init__(self, n_components):
            self.n_components = n_components
            self.linear_discriminants = None

        def fit(self, X, y):
            n_features = X.shape[1]     #berikan jumlah baris dalam array
            class_labels = np.unique(y) #dapatkan nilai unik dari array yang diberikan sebagai parameter

            # Scatter matrix dalam kelas:
            # SW = sum((X_c - mean_X_c)^2 )

            # Scatter matrix antar kelas:
            # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

            mean_overall = np.mean(X, axis=0)
            SW = np.zeros((n_features, n_features))
            SB = np.zeros((n_features, n_features))
            for c in class_labels:
                X_c = X[y == c]
                mean_c = np.mean(X_c, axis=0)
                # (16, n_c) * (n_c, 16) = (16,16) -> transpose
                SW += (X_c - mean_c).T.dot((X_c - mean_c))

                # (16, 1) * (1, 16) = (16,16) -> reshape
                n_c = X_c.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
                SB += n_c * (mean_diff).dot(mean_diff.T)

            # Menghitung SW^-1 * SB
            A = np.linalg.inv(SW).dot(SB)
            # Mencari nilai eigen dan vektor eigen dari SW^-1 * SB
            eigenvalues, eigenvectors = np.linalg.eig(A)
            # -> vektor eigen v = [:,i] vektor kolom, transpose untuk mempermudah kalkulasi
            # urutan vektor eigen dari tinggi ke rendah
            eigenvectors = eigenvectors.T
            idxs = np.argsort(abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[idxs]
            # simpan vektor eigen dengan nilai eigen terbesar
            self.linear_discriminants = eigenvectors[0:self.n_components]

        def transform(self, X):
            # project data
            return np.dot(X, self.linear_discriminants.T)

    #Terapkan LDA ke model
    X5 = df5.iloc[:,:-1]
    y5 = df5.iloc[:,-1]
    y5 = y5.astype('int')

    #proyeksi data ke bentuk 1 lD utama
    lda = LDA(1)
    lda.fit(X5.values,y5.values)
    X5_projected = lda.transform(X5)

    #Hasil LDA
    X5_lda = X5_projected.real
    df5['LDA'] = X5_lda
    LDA_result = pd.DataFrame(df5['LDA'])

    df6 = pd.concat([LDA_result, y], axis=1)
    df6 = df6.round(decimals=3)
    X6 = df6.iloc[:,:-1].values
    y6 = df6.iloc[:,-1].values
    n6 = len(df6)

    #### PROSES SVM ####
    # Perhitungan SVM+IG+LDA
    waktu_mulai = time.time()

    # Pembagian data dengan k-fold cross validation dimana k=14
    num_folds = 14
    kfold = KFold(n_splits=num_folds)

    #Terapkan model
    model = svm.SVC(kernel='rbf', C=1000, gamma=0.5) 
    akurasi = cross_val_score(model, X6, y6.ravel(), cv=kfold)
    rerata_akurasi = akurasi.mean()
    hasil = np.around(rerata_akurasi*100, decimals=2)

    waktu_selesai = time.time()
    durasi = waktu_selesai-waktu_mulai
    durasi = np.around(durasi, decimals=2)

     # Perhitungan SVM
    waktu_mulai2 = time.time()
    df = pd.read_csv(data_file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    # Pembagian data dengan k-fold cross validation dimana k=14
    num_folds2 = 14
    kfold2 = KFold(n_splits=num_folds2)

    #Terapkan model
    model2 = svm.SVC(kernel='rbf', C=1000, gamma=0.5) 
    akurasi2 = cross_val_score(model2, X, y.ravel(), cv=kfold2)
    rerata_akurasi2 = akurasi2.mean()
    hasil2 = np.around(rerata_akurasi2*100, decimals=2)

    waktu_selesai2 = time.time()
    durasi2 = waktu_selesai2-waktu_mulai2
    durasi2 = np.around(durasi2, decimals=2)

    return render_template('demonstration.html', 
                            df=df,X=X,y=y,n=n,
                            df2=df2,X2=X2,y2=y2,n2=n2,t=threshold,
                            df3=df3,X3=X3,y3=y3,n3=n3,
                            df6=df6,X6=X6,y6=y6,n6=n6,
                            hasil=hasil,durasi=durasi,hasil2=hasil2,durasi2=durasi2)


# ======== Calculation ======================================================= #
# -------- Original Dataset -------------------------------------------------- #
@app.route('/original-dataset.html') 
def original_dataset():
    import pandas as pd 
    df=pd.read_csv('dataset\crawling.csv')

    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    n=len(df)
    return render_template('original-dataset.html',df=df,X=X,y=y,n=n)

# -------- Handled Dataset -------------------------------------------------- #
@app.route('/handled-dataset.html') 
def handled_dataset():
    import pandas as pd
    df=pd.read_csv('dataset\preprocessing.csv')

    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    n=len(df)
    return render_template('handled-dataset.html',df=df,X=X,y=y,n=n)


# -------- IG ---------------------------------------------------------------- #
@app.route('/ig.html')
def ig_dataset():
    import pandas as pd
    import numpy as np

    df=pd.read_csv('dataset\preprocessing.csv')
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    # entropy formula
    def hitung_impurity(X, kriteria_impurity):
        
        probs = X.value_counts(normalize=True)
        
        if kriteria_impurity == 'entropi':
            impurity = -1 * np.sum(np.log2(probs) * probs)
        else:
            raise ValueError('Kriteria impurity tidak diketahui')
            
        return(round(impurity, 3))

    # Gain formula
    def hitung_information_gain_X(df, y, atribut, kriteria):
                
        entropi_y = hitung_impurity(df[y], kriteria)

        list_entropi = list()
        list_bobot = list()

        for value in df[atribut].unique():
            nilai_df_X = df[df[atribut] == value]
            nilai_entropi = hitung_impurity(nilai_df_X [y], kriteria)
            list_entropi.append(round(nilai_entropi, 3))
            nilai_bobot = len(nilai_df_X ) / len(df)
            list_bobot.append(round(nilai_bobot, 3))

        sisa_impurity_X = np.sum(np.array(list_entropi) * np.array(list_bobot))
        information_gain = entropi_y - sisa_impurity_X
        information_gain=np.around(information_gain, decimals=3)
        return(atribut, information_gain)

    IG = []

    kriteria = 'entropi'
    for X in df.drop(columns='Label').columns:
        info_gain_X = hitung_information_gain_X(df, 'Label', X, kriteria)
        IG.append(info_gain_X)
        
    # Data of Gain with threshold
    df2 = pd.DataFrame(IG, columns = ['Atribut','Gain'])
    threshold = 0.02

    # Retrive dataframe of Gain
    columns = []
    for index, row in df2.iterrows():
        if row['Gain'] > threshold:
            x = row['Atribut']
            columns.append(x)

    df3 = pd.concat([df[columns], y], axis=1)
    df3 = df3.round(decimals=3)
    X3 = df3.iloc[:,:-1].values
    y3 = df3.iloc[:,-1].values
    n3 = len(df3)

    X2 = np.array(df2['Atribut']) 
    y2 = np.array(df2['Gain']) 
    n2 = len(df2)
    return render_template('ig.html',df2=df2,X2=X2,y2=y2,n2=n2,t=threshold,df3=df3,X3=X3,y3=y3,n3=n3)

# -------- LDA --------------------------------------------------------------- #
@app.route('/lda.html')
def lda_dataset():
    import pandas as pd
    import numpy as np
    from scipy.stats import zscore

    df=pd.read_csv('dataset\ig.csv')

    # Logarithm Transformation
    column=["Followers","Followings","Public member", "Likes","Total tweets"]
    df[column]=df[column].applymap(lambda x: np.log(x) if x > 0 else np.log(x+1))

    # Z-score
    df2=df.apply(zscore)

    # LDA model
    class LDA:

        # Objek __init__  menerima konstanta sebagai argumen
        def __init__(self, n_components):
            self.n_components = n_components
            self.linear_discriminants = None

        def fit(self, X, y):
            n_atribut = X.shape[1]     #berikan jumlah baris dalam array
            kelas_target = np.unique(y) #dapatkan nilai unik dari array yang diberikan sebagai parameter

            # Scatter matrix dalam kelas:
            # SW = sum((X_i - rerata_X_i)^2 )

            # Scatter matrix antar kelas:
            # SB = sum( n_X_i * (rerata_X_i - rerata_X)^2 )

            rerata_X = np.mean(X, axis=0)
            SW = np.zeros((n_atribut, n_atribut))
            SB = np.zeros((n_atribut, n_atribut))
            for i in kelas_target:
                X_i = X[y == i]
                rerata_X_i = np.mean(X_i, axis=0)
                # (16, n_X_i) * (n_X_i, 16) = (16,16) -> transpose
                SW += (X_i - rerata_X_i).T.dot((X_i - rerata_X_i))

                # (16, 1) * (1, 16) = (16,16) -> reshape
                n_X_i = X_i.shape[0]
                selisih_rerata = (rerata_X_i - rerata_X).reshape(n_atribut, 1)
                SB += n_X_i * (selisih_rerata).dot(selisih_rerata.T)

            # Menghitung SW^-1 * SB
            A = np.linalg.inv(SW).dot(SB)
            # Mencari nilai eigen dan vektor eigen dari SW^-1 * SB
            eigenvalues, eigenvectors = np.linalg.eig(A)
            # -> vektor eigen v = [:,i] vektor kolom, transpose untuk mempermudah kalkulasi
            # urutan vektor eigen dari tinggi ke rendah
            eigenvectors = eigenvectors.T
            hasil_sorting = np.argsort(abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[hasil_sorting]
            eigenvectors = eigenvectors[hasil_sorting ]
            # simpan vektor eigen dengan nilai eigen terbesar
            self.linear_discriminants = eigenvectors[0:self.n_components]

        def transform(self, X):
            # project data
            return np.dot(X, self.linear_discriminants.T)

    #Terapkan LDA ke model
    X2=df2.iloc[:,:-1]
    y2=df2.iloc[:,-1]
    y2=y2.astype('int')

    #proyeksi data ke bentuk 1 lD utama
    lda = LDA(1)
    lda.fit(X2.values,y2.values)
    X2_projected = lda.transform(X2)

    #Hasil LDA
    X2_lda =X2_projected.real
    df2['LDA'] = X2_lda
    LDA_result = pd.DataFrame(df2['LDA'])

    df3=pd.concat([LDA_result, y2], axis=1)
    df3=df3.round(decimals=3)
    X3=df3.iloc[:,:-1].values
    y3=df3.iloc[:,-1].values
    n3=len(df3)
    return render_template('lda.html',df=df,df2=df2,df3=df3,X3=X3,y3=y3,n3=n3)

# -------- SVM --------------------------------------------------------------- #
@app.route('/svm.html') 
def svm_dataset():
    import pandas as pd 
    import numpy as np
    from sklearn import svm
    from sklearn.model_selection import KFold, cross_val_score
    import time as time

    # Perhitungan SVM+IG+LDA
    waktu_mulai = time.time()
    df=pd.read_csv('dataset\lda.csv')
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1:].values

    # Pembagian data dengan k-fold cross validation dimana k=14
    num_folds = 14
    kfold = KFold(n_splits=num_folds)

    #Terapkan model
    model = svm.SVC(kernel='rbf', C=1000, gamma=0.5) 
    akurasi = cross_val_score(model, X, y.ravel(), cv=kfold)
    rerata_akurasi = akurasi.mean()
    hasil = np.around(rerata_akurasi*100, decimals=2)

    waktu_selesai = time.time()
    durasi = waktu_selesai-waktu_mulai
    durasi = np.around(durasi, decimals=2)

     # Perhitungan SVM
    waktu_mulai2 = time.time()

    # Pembagian data dengan k-fold cross validation dimana k=14
    num_folds2 = 14
    kfold2 = KFold(n_splits=num_folds2)
    df2=pd.read_csv('dataset\preprocessing.csv')
    X2=df2.iloc[:,:-1].values
    y2=df2.iloc[:,-1:].values

    #Terapkan model
    model2 = svm.SVC(kernel='rbf', C=1000, gamma=0.5) 
    akurasi2 = cross_val_score(model2, X2, y2.ravel(), cv=kfold2)
    rerata_akurasi2 = akurasi2.mean()
    hasil2 = np.around(rerata_akurasi2*100, decimals=2)

    waktu_selesai2 = time.time()
    durasi2 = waktu_selesai2-waktu_mulai2
    durasi2 = np.around(durasi2, decimals=2)
    return render_template('svm.html',hasil=hasil,hasil2=hasil2,durasi=durasi,durasi2=durasi2)


# -------- About ------------------------------------------------------------- #
@app.route("/about")
def about():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        user = helpers.get_user()
        return render_template('about.html', user=user)
    
# ======== Main ============================================================== #
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, host="0.0.0.0")
    