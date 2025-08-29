import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# ================================
# ✅ PAGE CONFIG
# ================================
st.set_page_config(page_title="🍊 Food Recommender", layout="wide")

# ================================
# ✅ CUSTOM ORANGE THEME (NON-DASHBOARD STYLE)
# ================================
st.markdown("""
    <style>
        body {
            background-color: #fff8f0;
            font-family: 'Poppins', sans-serif;
        }
        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: #FF7043;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }
        .input-section {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 600px;
            margin-bottom: 30px;
        }
        /* Text Input & Dropdown */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            border: 2px solid #FF7043;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
        }
        /* Button */
        .stButton>button {
            background-color: #FF7043;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 12px 30px;
            border: none;
            margin-top: 15px;
        }
        .stButton>button:hover {
            background-color: #E64A19;
        }
        /* Card for Recommendations */
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: left;
        }
        .card h4 {
            color: #FF7043;
            margin-bottom: 10px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #999;
        }
    </style>
""", unsafe_allow_html=True)

# ================================
# ✅ HEADER
# ================================
st.markdown('<div class="container"><div class="main-title">🍊 Food Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover dishes you’ll love, based on your taste!</div>', unsafe_allow_html=True)

# ================================
# ✅ LOAD DATA
# ================================
food_data = pd.read_excel('food_data.xlsx')
food_data.columns = food_data.columns.str.strip().str.lower().str.replace(' ', '_')
food_data.fillna('', inplace=True)

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
# ✅ INPUT SECTION (Centered)
# ================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)
food_name = st.text_input("🍽️ Enter your favorite food:")
filtering_method = st.selectbox("Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])
st.markdown('</div>', unsafe_allow_html=True)

# ================================
# ✅ SHOW RESULTS (Centered)
# ================================
if food_name:
    closest_match = find_closest_match(food_name)

    if closest_match:
        st.success(f"✅ Closest match found: {closest_match}")

        food_details = food_data[food_data['food_name'] == closest_match].iloc[0]
        st.markdown(f"""
            <div class="card">
                <h4>{closest_match}</h4>
                <p><b>Category:</b> {food_details['category']}</p>
                <p><b>Description:</b> {food_details['description'][:200]}...</p>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("🔥 Top 10 Recommendations")
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

# ================================
# ✅ FOOTER
# ================================
st.markdown('<div class="footer">🍊 Developed with Streamlit | Food Recommender System</div>', unsafe_allow_html=True)
