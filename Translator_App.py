import streamlit as st
import os
from datasets import load_dataset
from transformers import pipeline
import string

# Define the list of available languages
languages = [
    "English",
    "French",
    "German",
    "Romanian"
]

def remove_punctuation(text):
    return "".join(char for char in text if char not in string.punctuation)

def translate_text(text, source_language, target_language):
    t5_small_pipeline = pipeline(
    task="text2text-generation",
    model="t5-large",
    max_length=1000,
    model_kwargs={"cache_dir": './Translate/t5_large' },
    )

    prompt=f"translate {source_language} to {target_language}: {text}"

    translation = t5_small_pipeline(prompt)
    return translation

# Streamlit web app
def main():
    st.title("Translate My Text")

    # Input text
    text = st.text_area("Enter the text to translate")
    text=remove_punctuation(text)

    # Source language dropdown
    source_language = st.selectbox("Select source language", languages)

    # Target language dropdown
    target_language = st.selectbox("Select target language", languages)

    # Translate button
    if st.button("Translate"):
        translation = translate_text(text, source_language, target_language)[0]['generated_text']
        st.markdown(f'<p style="color: blue; font-size: 25px;"> {translation}</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
