# Libreries
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googleapiclient.discovery import build
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import isodate

load_dotenv()

# Load youtube API key
youtude_api = os.getenv("YOUTUBE_API_KEY") 


# Importent variables
MAX_RESULTS = 10

# Function to search for videos on YouTube
def search_youtube_videos(query, max_results=5):
    # youtube_api = os.getenv("YOUTUBE_API_KEY")    
    try:
        youtube = build("youtube", "v3", developerKey=youtude_api)
        response = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=max_results,
            type=["video", "playlist"],
        ).execute()

        # print(response)

        videos = []
        for item in response.get("items", []):
            if item["id"]["kind"] != "youtube#video":
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]
                playlist_id = item["id"]["playlistId"]
                url = f"https://www.youtube.com/playlist?list={playlist_id}"
                videos.append({
                    "title": title,
                    "description": description,
                    "url": url
                })
            else:
                id = item["id"]["videoId"]
                video_data = get_video_details(id)
                if video_data:
                    videos.append(video_data)
                
        return videos
    except Exception as e:
        return f"Error: {e}"

def get_video_details(video_id):
    youtube = build('youtube', 'v3', developerKey=youtude_api)
    
    response = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    ).execute()

    if not response['items']:
        return None

    video = response['items'][0]
    
    data = {
        'title': video ['snippet']['title'],
        'description': video['snippet']['description'],
        'url': f"https://www.youtube.com/watch?v={video_id}",
        'views': int(video['statistics'].get('viewCount', 0)),
        'likes': int(video['statistics'].get('likeCount', 0)),
        'duration': video['contentDetails']['duration']
    } 
    data['duration'] = isodate.parse_duration(data['duration']).total_seconds()
    if data['duration'] >= 300:
        return data
    else:
        return None

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

User message: "{user_query}"

From the following videos, recommend the top 3 that are most relevant and useful, and most likely to be watched by the user from the following list:

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
    videos = search_youtube_videos(user_query, MAX_RESULTS)
    if type(videos) == list: 
        formatted_list = formate_videos_metadata(videos)
        prompt_template = prompt(user_query, formatted_list)
        try:
            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0.5,
                max_retries=3 
            )

            model = LLMChain(llm=llm, prompt=prompt_template)
            response = model.invoke({
                "user_query": user_query,
                "video_metadata_list": formatted_list
            })

            return response["text"]
        except Exception as e:
            return f"Error: {e}"
    else:
        return f"Error: {videos}"

# Main Code
def main():
    SEARCH_QUERY = input("Enter what do you want to search: ")
    response = recommend_videos_with_llm(SEARCH_QUERY)

    print("\nRecommended Videos:\n")
    print(response)

if __name__ == "__main__":
    main()