"""
Authors: Josef LaFranchise, Anirudha Shastri, Elio Khouri, Karthik Koduru
Date: 11/22/2024
CS 7180: Advanced Perception
assitant.py
"""

confidence_threshold = 0.6

""" Produces a response from a template based on the given object and its distance in the scene.

    Args:
     - object_name : name of the object 
     - distance : distance of the object from the camera
     - confidence : confidence level that the object is classified correctly

"""
def generte_object_reponse(object_name, distance, confidence):
    response = ""
    article = "a"
    
    if confidence < confidence_threshold:
        object_name = "object"
        
    if object_name[0] in {"a","e","i","o","u"}:
        article = "an"
        
    response = f"There is {article} {object_name} {distance:.2f} meters infront of you"
    
    print(response)
    
        
    