from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from model import ASLClassifier

import copy
import itertools
import csv
import json


ASL_CLASSIFIER_LABEL = 'model/asl_classifier/asl_labels.csv'

# Read ASL Labels
with open(ASL_CLASSIFIER_LABEL, encoding='utf-8-sig') as f:
    asl_classifier_labels = csv.reader(f)
    asl_classifier_labels = [
        row[0] for row in asl_classifier_labels
    ]

# ASL Classifier Model
asl_classifier = ASLClassifier()

app = FastAPI()
active_connections = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main():
    return {"message": "Hand Gesture Recognition with MediaPipe and OpenCV"}


def pre_process_landmark(landmark_array):
    temp_landmark_array = copy.deepcopy(landmark_array)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark in enumerate(landmark_array):
        if index == 0:
            base_x, base_y = landmark[0], landmark[1]
        
        temp_landmark_array[index][0] -= base_x
        temp_landmark_array[index][1] -= base_y

    # Convert to a one-dimensional list
    temp_landmark_array = list(itertools.chain.from_iterable(temp_landmark_array))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_array)))

    def normalize_(n):
        return n / max_value
    
    temp_landmark_array = list(map(normalize_, temp_landmark_array))

    return temp_landmark_array


@app.websocket("/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            data = await websocket.receive_json()
            landmark_list = data.get("landmark_list", [])
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            asl_sign_id = asl_classifier(pre_processed_landmark_list)
            asl_sign_label = asl_classifier_labels[asl_sign_id]
            response = {"asl_sign_label": asl_sign_label}
            await websocket.send_json(response)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(active_connections)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
