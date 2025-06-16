"""Run Movie Recommendation System by weeks"""

import streamlit as st

st.set_page_config(page_title="Movie Rec Weeks", page_icon="🎬")

week = st.sidebar.selectbox("Choose Week", [
    "Week 1: Data Collection",
    "Week 2: Collaborative Filtering", 
    "Week 3: Content-Based & Hybrid",
    "Week 4: Evaluation & Interface"
])

if week.startswith("Week 1"):
    import week1_movie_data
    week1_movie_data.main()
elif week.startswith("Week 2"):
    import week2_collaborative
    week2_collaborative.main()
elif week.startswith("Week 3"):
    import week3_hybrid
    week3_hybrid.main()
elif week.startswith("Week 4"):
    import week4_interface
    week4_interface.main()
