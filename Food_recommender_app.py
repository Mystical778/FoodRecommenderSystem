import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# Load food dataset
food_data = pd.read_excel('food_data.xlsx')  # Load your food dataset

# Ensure column names are clean
food_data.columns = food_data.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing values
food_data.fillna('', inplace=True)

# Combine numeric data for collaborative filtering (e.g., description length)
food_data['description_length'] = food_data['description'].apply(len)  # Example feature for CF
numeric_data = food_data[['description_length']].fillna(0)

# Content features for content-based filtering
food_data['combined_features'] = (
    food_data['category'].astype(str) + ' ' +
    food_data['description'].astype(str)
)

# Function to find the closest match
def find_closest_match(user_input):
    food_names = food_data['food_name'].tolist()
    closest_matches = df.get_close_matches(user_input, food_names, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

# Collaborative Filtering
def collaborative_filtering(food_name):
    similarity_matrix = cosine_similarity(numeric_data)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_matrix[food_index]))
    sorted_similar_foods = sorted(similar_foods, key=lambda x: x[1], reverse=True)
    return sorted_similar_foods[1:11]  # Exclude itself

# Content-Based Filtering
def content_based_filtering(food_name):
    cv = CountVectorizer()
    features_matrix = cv.fit_transform(food_data['combined_features'])
    similarity_scores = cosine_similarity(features_matrix, features_matrix)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_scores[food_index]))
    sorted_similar_foods = sorted(similar_foods, key=lambda x: x[1], reverse=True)
    return sorted_similar_foods[1:11]  # Exclude itself

# Streamlit UI
st.title("🍔 Food Recommender System")

# User input
food_name = st.text_input("Enter the name of a food:")

# Check for empty input
if food_name:
    closest_match = find_closest_match(food_name)

    if closest_match:
        st.success(f"Closest match found: {closest_match}")

        # Display the food details for the closest match
        food_details = food_data[food_data['food_name'] == closest_match].iloc[0]
        st.subheader(f"Details for '{closest_match}':")
        st.write(f"**Category:** {food_details['category']}")
        st.write(f"**Description:** {food_details['description'][:200]}...")  # Display a portion of the description
        
        # Recommendations based on the selected method
        filtering_method = st.selectbox("Select Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])
        
        if filtering_method == "Collaborative Filtering":
            recommendations = collaborative_filtering(closest_match)
            st.subheader(f"Top 10 foods similar to '{closest_match}' (Collaborative Filtering):")
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.write(f"- {recommended_food['food_name']} (Score: {food[1]:.4f})")

        elif filtering_method == "Content-Based Filtering":
            recommendations = content_based_filtering(closest_match)
            st.subheader(f"Top 10 foods similar to '{closest_match}' (Content-Based Filtering):")
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.write(f"- {recommended_food['food_name']} (Score: {food[1]:.4f})")

    else:
        st.warning(f"No close match found for: '{food_name}'")
else:
    st.info("Please enter a food name to get recommendations.")
