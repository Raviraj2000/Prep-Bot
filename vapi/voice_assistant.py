import os
import requests

# Define the VAPI endpoint
VAPI_ENDPOINT = "https://api.vapi.ai/tts"

def text_to_speech(text: str):
    try:
        # Validate API Key
        api_key = os.environ.get('VAPI_API_KEY')
        if not api_key:
            raise ValueError("VAPI_API_KEY is not set in the environment variables.")
        
        # Prepare headers and payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice": "en-US-Wavenet-D",  # Replace with your preferred voice ID
            "format": "mp3"
        }
        
        # Make the API request
        response = requests.post(VAPI_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        audio_url = data.get("audio_url")
        
        if not audio_url:
            raise ValueError("No 'audio_url' found in the API response.")
        
        return audio_url
    
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error: {req_err}")
        return None
    
    except ValueError as val_err:
        print(f"Value Error: {val_err}")
        return None
    
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None
