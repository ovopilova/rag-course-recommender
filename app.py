import streamlit as st
import openai
import faiss
import requests
import json
import os
from typing import List
from openai.embeddings_utils import get_embedding

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")  # Замените на ваш ключ

# Step 1: Load course data (Mockup for now, replace with actual parsing)
def load_course_data():
    return [
        {"title": "Data Science с нуля", "description": "Курс для начинающих изучать Data Science."},
        {"title": "Анализ данных в Python", "description": "Подробное изучение анализа данных с использованием Python."},
        {"title": "Машинное обучение", "description": "Основы машинного обучения: теория и практика."},
        {"title": "Продвинутый SQL", "description": "Углубленный курс по работе с базами данных."},
        {"title": "Нейронные сети", "description": "Создание и обучение нейронных сетей на практике."},
    ]

# Step 2: Generate embeddings for courses
def generate_embeddings(course_data: List[dict]):
    course_descriptions = [course["description"] for course in course_data]
    embeddings = [
        get_embedding(description, engine="text-embedding-ada-002")
        for description in course_descriptions
    ]
    return embeddings

# Step 3: Build FAISS index
def build_faiss_index(embeddings):
    dimension = len(embeddings[0])  # Dimension of embeddings
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

# Step 4: Find the best course match
def find_best_match(user_query, course_data, index):
    query_embedding = get_embedding(user_query, engine="text-embedding-ada-002")
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k=1)
    return course_data[indices[0][0]], distances[0][0]

# Streamlit App
st.title("Рекомендатор курсов")
st.write("Введите запрос, чтобы найти подходящий курс на Karpov.Courses")

# Load and process course data
course_data = load_course_data()
embeddings = generate_embeddings(course_data)
index = build_faiss_index(embeddings)

# User input
user_query = st.text_input("Опишите, что вы хотите изучить:")
if user_query:
    best_match, distance = find_best_match(user_query, course_data, index)
    st.write("Рекомендуемый курс:")
    st.subheader(best_match["title"])
    st.write(best_match["description"])
    st.write(f"Релевантность: {1 - distance:.2f}")
