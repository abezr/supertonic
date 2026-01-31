from gradio_client import Client

try:
    client = Client("patriotyk/styletts2-ukrainian")
    print("API Connected!")
    client.view_api()
except Exception as e:
    print(f"Error accessing API: {e}")
