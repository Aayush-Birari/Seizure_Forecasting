import os
import datetime
import time
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# The strict privacy scope we configured in Google Cloud
SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read']

def authenticate_google_fit():
    """Handles OAuth2 login and saves a secure token so the user doesn't login every day."""
    creds = None
    
    # If the user logged in before, load their saved session
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
        # Force Google to issue a permanent refresh token
        creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return creds

def fetch_todays_activity():
    """Fetches today's total steps and estimates distance for maximum privacy."""
    try:
        creds = authenticate_google_fit()
        fitness_service = build('fitness', 'v1', credentials=creds)

        # Get midnight of today in milliseconds
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time_millis = int(today.timestamp() * 1000)
        end_time_millis = int(time.time() * 1000)

        # 1. Build the exact request for Steps ONLY (No GPS location scope required)
        request_body = {
            "aggregateBy": [
                {"dataTypeName": "com.google.step_count.delta"}
            ],
            "bucketByTime": {"durationMillis": 86400000}, # 1 day bucket
            "startTimeMillis": start_time_millis,
            "endTimeMillis": end_time_millis
        }

        response = fitness_service.users().dataset().aggregate(userId="me", body=request_body).execute()
        
        steps = 0

        # 2. Parse the nested Google API response
        for bucket in response.get('bucket', []):
            for dataset in bucket.get('dataset', []):
                for point in dataset.get('point', []):
                    for value in point.get('value', []):
                        if 'intVal' in value:
                            steps += value['intVal']

        # 3. Privacy-First Distance Estimation (Average human stride is ~0.762m)
        distance_km = steps * 0.000762

        return steps, round(distance_km, 2)
        
    except Exception as e:
        print(f"Google Fit API Error: {e}")
        return 0, 0.0