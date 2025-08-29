import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# Load food dataset
food_data = pd.read_excel('food_data.xlsx')

# Ensure column names are clean
food_data.columns = food_data.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing values
food_data.fillna('', inplace=True)

# Combine numeric data for collaborative filtering
food_data['description_length'] = food_data['description'].apply(len)
numeric_data = food_data[['description_length']].fillna(0)

# Content features
food_data['combined_features'] = (
    food_data['category'].astype(str) + ' ' +
    food_data['description'].astype(str)
)

# CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #0f1116;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        color: #ff6600;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #bbb;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .stTextInput>div>div>input {
        background: #1b1e24;
        color: #fff;
        border: 2px solid #ff6600;
        border-radius: 8px;
        padding: 12px;
    }
    .stSelectbox>div>div>select {
        background: #1b1e24;
        color: #fff;
        border: 2px solid #ff6600;
        border-radius: 8px;
        padding: 8px;
    }
    .food-card {
        background-color: #1b1e24;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .food-card h2 {
        color: #ff6600;
    }
    .recommendation-item {
        background: #292d36;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .recommendation-item:hover {
        background: #ff6600;
        color: #000;
    }
    </style>
""", unsafe_allow_html=True)

# Functions
def find_closest_match(user_input):
    food_names = food_data['food_name'].tolist()
    closest_matches = df.get_close_matches(user_input, food_names, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

def collaborative_filtering(food_name):
    similarity_matrix = cosine_similarity(numeric_data)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_matrix[food_index]))
    sorted_similar_foods = sorted(similar_foods, key=lambda x: x[1], reverse=True)
    return sorted_similar_foods[1:11]

def content_based_filtering(food_name):
    cv = CountVectorizer()
    features_matrix = cv.fit_transform(food_data['combined_features'])
    similarity_scores = cosine_similarity(features_matrix, features_matrix)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_scores[food_index]))
    sorted_similar_foods = sorted(similar_foods, key=lambda x: x[1], reverse=True)
    return sorted_similar_foods[1:11]

# UI
st.markdown('<h1 class="title">🍔 Food Recommender System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find similar foods based on your choice</p>', unsafe_allow_html=True)

food_name = st.text_input("Enter the name of a food:")

if food_name:
    closest_match = find_closest_match(food_name)

    if closest_match:
        st.success(f"Closest match found: {closest_match}")

        # Display food details
        food_details = food_data[food_data['food_name'] == closest_match].iloc[0]
        st.markdown(f"""
        <div class="food-card">
            <h2>{closest_match}</h2>
            <p><strong>Category:</strong> {food_details['category']}</p>
            <p><strong>Description:</strong> {food_details['description'][:200]}...</p>
        </div>
        """, unsafe_allow_html=True)

        filtering_method = st.selectbox("Select Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])

        if filtering_method == "Collaborative Filtering":
            recommendations = collaborative_filtering(closest_match)
            st.subheader(f"Top 10 foods similar to '{closest_match}' (Collaborative Filtering):")
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.markdown(f"<div class='recommendation-item'>{recommended_food['food_name']} (Score: {food[1]:.4f})</div>", unsafe_allow_html=True)

        elif filtering_method == "Content-Based Filtering":
            recommendations = content_based_filtering(closest_match)
            st.subheader(f"Top 10 foods similar to '{closest_match}' (Content-Based Filtering):")
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.markdown(f"<div class='recommendation-item'>{recommended_food['food_name']} (Score: {food[1]:.4f})</div>", unsafe_allow_html=True)

    else:
        st.warning(f"No close match found for: '{food_name}'")
else:
    st.info("Please enter a food name to get recommendations.")
