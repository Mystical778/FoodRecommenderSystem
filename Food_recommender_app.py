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
food_data['description_length'] = food_data['description'].apply(len)
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
    return sorted_similar_foods[1:11]

# Content-Based Filtering
def content_based_filtering(food_name):
    cv = CountVectorizer()
    features_matrix = cv.fit_transform(food_data['combined_features'])
    similarity_scores = cosine_similarity(features_matrix, features_matrix)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_scores[food_index]))
    sorted_similar_foods = sorted(similar_foods, key=lambda x: x[1], reverse=True)
    return sorted_similar_foods[1:11]

# Inject custom CSS for styling
# Inject custom CSS for dark theme with orange accents
st.markdown("""
    <style>
    /* Global Background & Text */
    body {
        background-color: #121212;
        color: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, sans-serif;
    }

    /* Main Title */
    .title {
        text-align: center;
        color: #ff6600;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #bbbbbb;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Text Input */
    .stTextInput > div > input {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border: 2px solid #ff6600 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* Select Box Fix (Dropdown + Text) */
    div[data-baseweb="select"] {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border: 2px solid #ff6600 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #ffffff !important;
        background-color: #1f1f1f !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ff6600 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        cursor: pointer !important;
    }
    .stButton>button:hover {
        background-color: #e65c00 !important;
    }

    /* Food Card */
    .food-card {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
        border-left: 6px solid #ff6600 !important;
    }
    .food-card h4 {
        color: #ff6600 !important;
        margin-bottom: 5px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #ff6600;
        border-radius: 4px;
    }

    /* Recommendation List */
    .recommendation-item {
        padding: 8px;
        margin-bottom: 6px;
        background: #1f1f1f;
        border-radius: 6px;
        border-left: 4px solid #ff6600;
        transition: background 0.3s ease;
    }
    .recommendation-item:hover {
        background: #292929;
    }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown('<div class="title">🍔 Food Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find delicious food recommendations based on your input!</div>', unsafe_allow_html=True)

# User input
food_name = st.text_input("Enter the name of a food:")

if food_name:
    closest_match = find_closest_match(food_name)

    if closest_match:
        st.success(f"Closest match found: {closest_match}")

        # Display food details
        food_details = food_data[food_data['food_name'] == closest_match].iloc[0]
        st.markdown(f"""
            <div class="food-card">
                <h4>{closest_match}</h4>
                <p><strong>Category:</strong> {food_details['category']}</p>
                <p><strong>Description:</strong> {food_details['description'][:200]}...</p>
            </div>
        """, unsafe_allow_html=True)

        filtering_method = st.selectbox("Select Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])

        if filtering_method == "Collaborative Filtering":
            recommendations = collaborative_filtering(closest_match)
            st.subheader(f"Top 10 similar foods (Collaborative Filtering):")
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.markdown(f"""
                    <div class="food-card">
                        <h4>{recommended_food['food_name']}</h4>
                        <p><strong>Category:</strong> {recommended_food['category']}</p>
                        <p><strong>Score:</strong> {food[1]:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)

        elif filtering_method == "Content-Based Filtering":
            recommendations = content_based_filtering(closest_match)
            st.subheader(f"Top 10 similar foods (Content-Based Filtering):")
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.markdown(f"""
                    <div class="food-card">
                        <h4>{recommended_food['food_name']}</h4>
                        <p><strong>Category:</strong> {recommended_food['category']}</p>
                        <p><strong>Score:</strong> {food[1]:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.warning(f"No close match found for: '{food_name}'")
else:
    st.info("Please enter a food name to get recommendations.")

