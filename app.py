from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json

app = Flask(__name__, static_folder='E:/Recom/public')
CORS(app)

# Load the dataset
df = pd.read_csv(('anime_with_synopsis.csv'))
animes_sys= pd.read_csv('anime_with_synopsis.csv')
animes_sys.dropna()

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Fill missing values in 'synopsis' column with empty strings
animes_sys['sypnopsis'] = animes_sys['sypnopsis'].fillna("")

# Compute TF-IDF matrix
matrix = tfidf.fit_transform(animes_sys['sypnopsis'])

# Calculate cosine similarity
cosine = linear_kernel(matrix, matrix)


# Create a Series with anime names as index and their corresponding indices as values
indices = pd.Series(animes_sys.index, index=animes_sys['Name']).drop_duplicates()

def cosine_to_percentage(cosine_similarity):
    return (cosine_similarity * 200)



@app.route('/search')
def search():
    query = request.args.get('query', '')
    if query:
        # Filter titles containing the query string
        results = df[df['Name'].str.contains(query, case=False, na=False)]
        # print(results.values.tolist()[0][2])
        results = results.fillna('')
    else:
        results=[]
    return jsonify(results.to_dict(orient='records'))

@app.route('/getrecom')
def getrecom():
    title = request.args.get('query', '')
    # title="Naruto: Shippuuden"
    idx = indices[title]
    sim_scores = enumerate(cosine[idx])
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[2:10]
    sim_indices = [i[0] for i in sim_scores]
    
    # Calculate percentage similarity and add it to the output
    sim_percentage = [cosine_to_percentage(i[1]) for i in sim_scores]
    print(sim_percentage)
    recommendations = animes_sys.iloc[sim_indices].copy()
    recommendations['Similarity'] = sim_percentage
    recommendations['alt'] = recommendations['alt'].fillna('')
    recommendations['src'] = recommendations['src'].fillna('')
    print(recommendations)
    # recommendations_with_percentage = recommendations.to_dict(orient='records')

    return jsonify(recommendations.to_dict(orient='records'))




# Serve the static files (like the React app)
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

# Serve the React app
@app.route('/')
def root():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)



