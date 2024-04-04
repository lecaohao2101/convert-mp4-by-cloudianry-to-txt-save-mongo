import cloudinary
import cloudinary.api
import requests
import os
import dotenv
from pymongo import MongoClient
import concurrent.futures
import whisper
import openai

dotenv.load_dotenv()

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
MONGO_URI = os.getenv("MONGO_URI")
openai.api_key = os.getenv("OPENAI_API_KEY")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

client = MongoClient(MONGO_URI)
db = client.transcripts
transcript_collection = db.transcripts

# Define functions
def list_videos():
    result = cloudinary.api.resources(
        resource_type="video",
        type="upload",
        max_results=500
    )
    return result['resources']

def download_video(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

def transcribe_and_save_video(url):
    filename = url.split('/')[-1]
    filepath = os.path.join("downloaded_videos", filename)

    download_video(url, filepath)
    print(f"Downloaded (temporarily): {filename}")

    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the video file
    transcript = model.transcribe(filepath)

    # Extract the transcript text
    transcript_text = transcript["text"]

    transcript_file_name = f"{os.path.splitext(filename)[0]}.txt"
    transcript_collection.update_one({"name": transcript_file_name}, {"$set": {"name": transcript_file_name, "text": transcript_text}}, upsert=True)
    print(f"Transcript for {filename} saved in MongoDB as {transcript_file_name}")

    os.remove(filepath)
    print(f"Deleted video file: {filename}")

def main():
    os.makedirs("downloaded_videos", exist_ok=True)

    all_videos = list_videos()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(transcribe_and_save_video, [video['secure_url'] for video in all_videos])

if __name__ == "__main__":
    main()
