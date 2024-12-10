import os
import pandas as pd
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
recipe_path = os.path.join(current_dir, 'recipe.csv')
#model_path = os.path.join(current_dir, 'model.h5')

recipes_df = pd.read_csv(recipe_path)
#model = tf.keras.models.load_model(model_path)

vectorizer = TfidfVectorizer()
recipes_vector = vectorizer.fit_transform(recipes_df['ingredients'])

def recommend_recipe_updated(input_ingredients, data, recipes_vector, weight_loves=0.5, weight_steps=0.3, weight_ingredients=0.2):
    """
    Recommends a recipe based on the given input ingredients, considering popularity, efficiency, and ingredient matching.
    """
    # Transform input ingredients into vector
    input_vector = vectorizer.transform([input_ingredients])

    # Calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, recipes_vector).flatten()

    # Normalize features
    # normalized_loves = data['Loves'] / data['Loves'].max()
    # normalized_steps = 1 - (data['Total Steps'] / data['Total Steps'].max())  # Minimize steps
    # normalized_ingredients = 1 - (data['Total Ingredients'] / data['Total Ingredients'].max())  # Minimize ingredients

    # Weighted scoring
    score = (
        similarity_scores
        # similarity_scores +
        # weight_loves * normalized_loves +
        # weight_steps * normalized_steps +
        # weight_ingredients * normalized_ingredients
    )

    # Get the top 3 recipes
    top_indices = np.argpartition(score, -3)[-3:]  # Get indices of top 3 scores
    top_recipes = data.iloc[top_indices]  # Get the corresponding recipes

    results = []
    for index in top_indices:
        results.append({
            'title': data.iloc[index]['title'],
            'ingredients': data.iloc[index]['ingredients'],
            'steps': data.iloc[index]['directions'],
            # 'Loves': data.iloc[index]['Loves'],
            # 'Total Steps': data.iloc[index]['Total Steps'],
            # 'Total Ingredients': data.iloc[index]['Total Ingredients'],
            'Score': score[index]
        })

    return results


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    input_ingredients = request.form['query']
    results = recommend_recipe_updated(input_ingredients, recipes_df, recipes_vector) 

    return jsonify({
        'results': [{
            'title': recipe['title'],
            'ingredients': recipe['ingredients'],
            'steps': recipe['steps'],
            # 'loves': recipe['Loves'],
            # 'total_steps': recipe['Total Steps'],
            # 'Total Ingredients': recipe['Total Ingredients'],
            } for recipe in results]
        })
