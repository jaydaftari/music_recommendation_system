from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process

# Load models and datasets
knn5Model = pickle.load(open('knn5Model.pk1', 'rb'))
recommendation_set = pd.read_csv('recommendation_set.csv')
music_data = pd.read_csv('music_data.csv')
X_test = pd.read_csv('X_test.csv')

#recommendation_set.drop(X_test.columns[0], axis=1, inplace=True)
X_test.drop(X_test.columns[0], axis=1, inplace=True)

# Define recommender
def recommender(song_name, data, model):
    match = process.extractOne(song_name, recommendation_set['song'])
    if match:
        idx = match[2]
        requiredSongs = recommendation_set.select_dtypes(np.number).drop(columns=['cat', 'cluster', 'year']).copy()
        distances, indices = model.kneighbors(requiredSongs.iloc[idx].values.reshape(1, -1))
        rec = []
        for i in indices[0][:5]:  # Get top 5 recommendations
            rec.append(f"{music_data['song'][i]} by {music_data['artist'][i]}")
        return rec
    else:
        return ["Song not found in dataset"]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/songs', methods=['POST'])
def songs():
    try:
        # Get user input
        song_name = request.form.get('song_name', '')
        # Fetch recommendations
        recommendations = recommender(song_name, X_test, knn5Model)
        # Render results
        return render_template('songs.html', song_name=song_name, recommendations=recommendations)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

