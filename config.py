import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_FILE = os.path.join(BASE_DIR, 'secrets.json')

def load_secrets():
    if not os.path.exists(SECRETS_FILE):
        raise FileNotFoundError(f"Secrets file not found at {SECRETS_FILE}. Please create it based on secrets_example.json")
    
    with open(SECRETS_FILE, 'r') as f:
        return json.load(f)

secrets = load_secrets()

API_URL = secrets.get('api_url')
SECRET_KEY = secrets.get('secret_key')
CAPTION_API_KEY = secrets.get('caption_api_key')
GOOGLE_APPLICATION_CREDENTIALS = secrets.get('GOOGLE_APPLICATION_CREDENTIALS')
CAPTION_URL_BASE = 'https://apis.openapi.sk.com/vision/v1/caption'
