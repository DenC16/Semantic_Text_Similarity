import streamlit as st

from STS import countvectorizer_cosine_distance_method

st.markdown("""
# Semantic Text Similarity
This web app allows you to check similarity between 2 sentences using machine learning.

**Credits**
- Web App is built using `Python` + `Streamlit` by [Dhyaneswaran](https://bit.ly/3wuZcbA) 
---
""")

with st.form(key = 'my_form_1'):
    text_1 = st.text_input(label = 'Sentence 1')
    submit_button = st.form_submit_button(label='Submit')

with st.form(key = 'my_form_2'):
    text_2 = st.text_input(label = 'Sentence 2')
    submit_button = st.form_submit_button(label = 'Submit')


if submit_button:
    sim_score = countvectorizer_cosine_distance_method(s1 = text_1, s2 = text_2)
    st.write("similarity score")
    st.text(sim_score)
