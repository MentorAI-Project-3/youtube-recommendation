# Libreries
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googleapiclient.discovery import build
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(".env")

# Load youtube API key
youtube_api = os.getenv("YOUTUBE_API_KEY") 
# print("Youtube API Key: ", youtube_api)

# Importent variables
MAX_RESULTS = 10

# Function to search for videos on YouTube
def search_youtube_videos(query, max_results=5, api_key=youtube_api):
    youtube = build("youtube", "v3", developerKey = youtube_api)
    response = youtube.search().list(
        q=query,
        part="snippet", # 
        maxResults=max_results,
        type="video"
    ).execute()

    videos = []
    for item in response.get("items", []):
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        video_id = item["id"]["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({
            "title": title,
            "description": description,
            "url": url
        })

    return videos

# Function to format video metadata
def formate_videos_metadata(videos):
    metadata = ""
    for idx, vid in enumerate(videos, start = 1):
        metadata += f"{idx}. Title: {vid['title']}\nDescription: {vid['description']}\n Link: {vid['url']}\n\n"
    
    return metadata

#Function to create a prompt for the LLM
def prompt(user_query, video_metadata_list):
    prompt_template = PromptTemplate(
        input_variables=["user_query", "video_metadata_list"],
        template= """
You are an intelligent assistant helping users find the best YouTube videos.
User query: "{user_query}"
From the following videos, recommend the top 3 that are most relevant and useful:
{video_metadata_list}

Give reasult including the title and link only in the following format:
1. Title: video title, Link: video link
2. Title: video title, Link: video link
3. Title: video title, Link: video link
"""
)
    return prompt_template

# Function to recommend videos using LLM
def recommend_videos_with_llm(user_query):
    videos = search_youtube_videos(user_query, MAX_RESULTS, youtube_api)
    formatted_list = formate_videos_metadata(videos)
    prompt_template = prompt(user_query, formatted_list)
    videos = search_youtube_videos(user_query, MAX_RESULTS)
    formatted_list = formate_videos_metadata(videos)

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.5
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.invoke({
        "user_query": user_query,
        "video_metadata_list": formatted_list
    })

    return response["text"]

# Main Code
def main():
    SEARCH_QUERY = input("Enter what do you want to search: ")
    response = recommend_videos_with_llm(SEARCH_QUERY)

    print("\nRecommended Videos:\n")
    print(response)

if __name__ == "__main__":
    main()