import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# ================================
# ✅ PAGE CONFIG
# ================================
st.set_page_config(page_title="Food Recommender Dashboard", layout="wide")

# ================================
# ✅ ORANGE THEME CSS
# ================================
st.markdown("""
    <style>
        .stApp {
            background-color: #fff8f0;
            font-family: 'Poppins', sans-serif;
        }
        /* Titles */
        .main-title {
            text-align: center;
            font-size: 42px;
            color: #FF7043; /* Warm Orange */
            font-weight: bold;
            margin-bottom: 5px;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #888;
            margin-bottom: 30px;
        }
        /* Text Input */
        .stTextInput>div>div>input {
            border: 2px solid #FF7043;
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
            background-color: #fff;
        }
        /* Dropdown */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #fff;
            color: #333;
            border-radius: 10px;
        }
        /* Buttons */
        .stButton>button {
            background-color: #FF7043;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #F4511E;
        }
        /* Cards */
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: 0.3s;
        }
        .card:hover {
            transform: scale(1.03);
        }
        .card h4 {
            color: #FF7043;
            margin-bottom: 5px;
        }
        .card p {
            font-size: 14px;
            margin: 2px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ================================
# ✅ HEADER
# ================================
st.markdown('<div class="main-title">🍔 Food Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover delicious recommendations based on your favorite food</div>', unsafe_allow_html=True)

# ================================
# ✅ LOAD DATA
# ================================
food_data = pd.read_excel('food_data.xlsx')
food_data.columns = food_data.columns.str.strip().str.lower().str.replace(' ', '_') 
food_data.fillna('', inplace=True)

# Feature engineering
food_data['description_length'] = food_data['description'].apply(len)
numeric_data = food_data[['description_length']].fillna(0)
food_data['combined_features'] = (
    food_data['category'].astype(str) + ' ' +
    food_data['description'].astype(str)
)

# ================================
# ✅ FUNCTIONS
# ================================
def find_closest_match(user_input):
    food_names = food_data['food_name'].tolist()
    closest_matches = df.get_close_matches(user_input, food_names, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

def collaborative_filtering(food_name):
    similarity_matrix = cosine_similarity(numeric_data)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_matrix[food_index]))
    return sorted(similar_foods, key=lambda x: x[1], reverse=True)[1:11]

def content_based_filtering(food_name):
    cv = CountVectorizer()
    features_matrix = cv.fit_transform(food_data['combined_features'])
    similarity_scores = cosine_similarity(features_matrix, features_matrix)
    food_index = food_data[food_data['food_name'] == food_name].index[0]
    similar_foods = list(enumerate(similarity_scores[food_index]))
    return sorted(similar_foods, key=lambda x: x[1], reverse=True)[1:11]

# ================================
# ✅ DASHBOARD LAYOUT
# ================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Search for a Food")
    food_name = st.text_input("Enter food name:")
    
    filtering_method = st.selectbox("Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])
    
with col2:
    if food_name:
        closest_match = find_closest_match(food_name)

        if closest_match:
            st.success(f"✅ Closest match found: {closest_match}")

            # Show details in a styled card
            food_details = food_data[food_data['food_name'] == closest_match].iloc[0]
            st.markdown(f"""
                <div class="card">
                    <h4>{closest_match}</h4>
                    <p><b>Category:</b> {food_details['category']}</p>
                    <p><b>Description:</b> {food_details['description'][:200]}...</p>
                </div>
            """, unsafe_allow_html=True)

            st.subheader(f"🔥 Top 10 Recommendations")
            recommendations = collaborative_filtering(closest_match) if filtering_method == "Collaborative Filtering" else content_based_filtering(closest_match)
            
            for food in recommendations:
                recommended_food = food_data.iloc[food[0]]
                st.markdown(f"""
                    <div class="card">
                        <h4>{recommended_food['food_name']}</h4>
                        <p><b>Category:</b> {recommended_food['category']}</p>
                        <p><b>Description:</b> {recommended_food['description'][:200]}...</p>
                        <p><b>Score:</b> {food[1]:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No close match found for: '{food_name}'")
    else:
        st.info("Please enter a food name to get recommendations.")
