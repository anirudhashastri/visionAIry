"""
Authors: Anirudha Shastri, Josef LaFranchise, Elio Khouri, Karthik Koduru
Date: 11/22/2024
CS 7180: Advanced Perception
LLMassistant.py
"""

def safely_stop_speaker(speaker):
    """
    Checks if the speaker is currently speaking and stops it if necessary.
    """
    if hasattr(speaker, "is_playing") and speaker.is_playing():
        print("Speaker is currently speaking. Stopping it.")
        speaker.stop()  # Stop the current speech
    else:
        print("Speaker is not speaking.")

def generate_llm_reponse(info_dict,speaker,client):
    """
    Generates a response from llama-3.1-8b-instant based on the information int the dictionary. Sends the 
    result to the speaker to play as TTS.
    
    Args:
     - info_dict : dictionary containing object names and distances
     - speaker : speaker that plays the TTS
     - client : groq API client
    """
            
    labels = []
    distances = []
    for key in info_dict:
        distance = info_dict[key]
        print(type(distance))
        print(f'{distance:.2f}')
        label_name = key.split()
        print(label_name[0])
        labels.append(label_name[0])
        distances.append(round(distance,1))

    # Prompt that is used to determine how the LLM describes the scene
    prompt = [
    {
        "role": "system",
        "content": """
        You are an AI assistant. Your task is to provide a concise response by listing each object with its corresponding distance 
        in a simple sentence. Do not add any explanations or commentary. Just state each object and its distance.All distances are in meters.
        """
    },
    {
        "role": "user",
        "content": f"""
        Here are two lists:
        Objects: {labels}
        Distances: {distances}
        """
    }
    ]
    try:
        response = client.chat.completions.create(
            messages=prompt,
            model="llama-3.1-8b-instant"
        )
        
        result = response.choices[0].message.content.strip()
        print(f"Generated completion : {result}")
        # Check and stop current TTS playback if needed
        safely_stop_speaker(speaker)
        
        speaker.say(result)
        
    except Exception as e:
            print(f"Failed to generate completion : {e}")
        

def call_generate_llm_response_in_thread(object_dict, speaker, client, threading):
    """
    Wrapper function to call `generate_llm_reponse` in a separate thread.
    
    Args:
     - object_dict : dictionary containing object names and distances
     - speaker : speaker that plays the audio
     - client : groq API client
     - threading : threading instance
    """

    thread = threading.Thread(target=generate_llm_reponse, args=(object_dict, speaker, client))
    thread.daemon = True  # Daemonize thread to ensure it ends with the main program
    thread.start()

    return True