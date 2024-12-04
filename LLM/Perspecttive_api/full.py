import requests
import json
import os  # For accessing environment variables

def analyze_text(text):
    # Retrieve the API key from an environment variable
    API_KEY = os.environ.get('PERSPECTIVE_API_KEY')
    if not API_KEY:
        print("API key not found in environment variables.")
        return

    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
           f'?key={API_KEY}')
    
    # The text to analyze
    data = {
        'comment': {'text': text},
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {}
        }
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Make the HTTP POST request to Perspective API
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        # Output the entire JSON response
        print("Full JSON Response:")
        print(json.dumps(result, indent=2))
        
        print("\nIndividual Attribute Scores:")
        attributes = result.get('attributeScores', {})
        for attribute, details in attributes.items():
            score = details.get('summaryScore', {}).get('value', 'N/A')
            print(f"- {attribute.replace('_', ' ').title()}: {score}")
    else:
        print(f'Error {response.status_code}: {response.text}')

if __name__ == "__main__":
    # Your text to analyze
    text_to_analyze = "I want to kill someone"
    analyze_text(text_to_analyze)