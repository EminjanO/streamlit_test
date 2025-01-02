import os
import streamlit as st
import openai
import pinecone

from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()

from pinecone import Pinecone
pc_api_key = os.getenv("PINECONE_API_KEY")

if not pc_api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

pc = Pinecone(api_key=pc_api_key)
index = pc.Index("movie-recommendation")


st.set_page_config(layout="wide")

st.title('OpenAI API Webapp')

st.sidebar.title("AI App")
applications = ["Blog Generation", "Generate Image", "Movie Recommendation"]
ai_app = st.sidebar.radio("Choose an AI App", applications)

def blog_genertion(topic, additional_pointers):
    prompt = f"""
    You are a copy writer with years of experience writing impactful blog that converge and help elevate brands.
    Your task is to write a blog on any topic system provides to you. Make sure to write in a format that works for Medium.
    Each blog should be separated into segments that have titles and subtitles. Each paragraph should be three sentences long.

    Topic: {topic}
    Additiona pointers: {additional_pointers}
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=1,
        max_tokens=700,
    )

    return response.choices[0].text.strip()

def generate_image(prompt, number_of_images=1):
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",
        n=number_of_images,
    )

    return response

if ai_app == "Blog Generation":
    st.header("Blog Generator")
    st.write("Input a topic to generate a blog post about it using OpenAI API.")
    input_text = st.text_area("Enter text here:")
    additional_pointers = st.text_area("Enter additional pointers here:")
    
    if st.button("Complete Text"):
        with st.spinner('Generating...'):
            completion = blog_genertion(input_text, additional_pointers)
            st.text_area("Generated blog:", value=completion, height=200)
elif ai_app == "Generate Image":
    st.header("Image Generator")
    st.write("Add a prompt to generate an image using OpenAI API and DALL-E model.")
    input_text = st.text_area("Enter text for image generation:")

    number_of_images = st.slider("Choose the number of images to generate", 1, 5, 1) 
    if st.button("Generate Image"):
        
        outputs = generate_image(input_text, number_of_images)
        for output in outputs.data:
            st.image(output.url)                        
elif ai_app == "Movie Recommendation":
    st.header("Movie Recommendor")
    st.write("Describe a movie you like and we will recommend you similar movies.")
    
    movie_description = st.text_area("Enter movie description:")

    if st.button("Get movies"):
        with st.spinner('Generating...'):
            vector = client.embeddings.create(
                model="text-embedding-ada-002",
                input=movie_description)

            result_vector = vector.data[0].embedding

            matches = index.query(
                vector=result_vector,
                top_k=10,
                include_metadata=True,
            )
            
            for match in matches['matches']:
                st.write(match['metadata']['title'])