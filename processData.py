from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.authoring.models import ApplicationCreateObject
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials
from functools import reduce

import json, time, uuid

authoringKey = 'aea93893196948b7a3fbdca2b6980ccd'
authoringEndpoint = 'https://westus.api.cognitive.microsoft.com/'
predictionKey = 'b3a7d6ce982f4d69a1179de71a4ec7e6'
predictionEndpoint = 'https://toneitclu.cognitiveservices.azure.com/'
app_id = '2579e3f1-2b88-4f22-947c-79143aa8764d'
app_name = 'ToneIt'
version_id = '0.1'

def utterance_with_MLEntity(intent, entity, sentence):
    label = {
        "text": sentence,
        "intentName": intent,
        "entitityLevels": [
            {
                "startCharIndex": 0,
                "endCharIndex": len(sentence) - 1,
                "entityName": entity
            }
        ]
    } 
    return label

def fillInUtterance(dataSet, client):
    for data in dataSet:
        utterance = utterance_with_MLEntity(data['tone'] + "Intent", data['tone'] + "Intent", data['sentence'])
        client.examples.add(app_id, version_id, utterance)
    
# SampleData = [
#     {
#         "sentence": "im feeling rather rotten so im not very ambitious right now",
#         "emotion": "sadness"  
#     },
#         {
#         "sentence": "im updating my blog because i feel shitty",
#         "emotion": "sadness"  
#     }
# ]

def checkEmotion():
    with open('sampleEmotion.json') as json_file:
        sample_data = json.load(json_file)
        emotion = set()
        for data in sample_data:
            emotion.add(data['tone'])
        print(emotion)

def makePrediction():
    runtimeCredentials = CognitiveServicesCredentials(predictionKey)
    clientRuntime = LUISRuntimeClient(endpoint=predictionEndpoint, credentials=runtimeCredentials)
    
    predictionRequest = { "query" : "I am so depressed" }

    predictionResponse = clientRuntime.prediction.get_slot_prediction(app_id, "Production", predictionRequest)
    print("Top intent: {}".format(predictionResponse.prediction.top_intent))
    print("Sentiment: {}".format (predictionResponse.prediction.sentiment))
    print("Intents: ")

    for intent in predictionResponse.prediction.intents:
        print("\t{}".format (json.dumps (intent)))
    print("Entities: {}".format (predictionResponse.prediction.entities))

def quickStart():
    client = LUISAuthoringClient(authoringEndpoint, CognitiveServicesCredentials(authoringKey))
    # with open('sampleEmotion.json') as json_file:
    #     sample_data = json.load(json_file)
    # fillInUtterance(sample_data, client)
    client.train.train_version(app_id, version_id)
    waiting = True
    while waiting:
        info = client.train.get_status(app_id, version_id)

        # get_status returns a list of training statuses, one for each model. Loop through them and make sure all are done.
        waiting = any(map(lambda x: 'Queued' == x.details.status or 'InProgress' == x.details.status, info))
        if waiting:
            print ("Waiting 10 seconds for training to complete...")
            time.sleep(10)
        else: 
            print ("trained")
            waiting = False
    client.apps.update_settings(app_id, is_public=True)

    responseEndpointInfo = client.apps.publish(app_id, version_id, is_staging=False)
    
    makePrediction()