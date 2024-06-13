import cv2
import numpy as np
import face_recognition
import logging
import time
from utils import GazeTracking
import gradio as gr

# Initialize Logger
logging.basicConfig(filename='./core/logging/pupil.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Load Face Data
usman_image = face_recognition.load_image_file("./core/data/Usman_Ahmed.jpg")
usman_face_encoding = face_recognition.face_encodings(usman_image)[0]

known_face_encodings = [usman_face_encoding]
known_face_names = ["Usman Ahmed"]

process_this_frame = True
alert_time = 3
time_not_focused = 0
last_logged_time = time.time()

gaze = GazeTracking()

def inference(frame):

    face_locations = []
    face_names = []

    global process_this_frame, time_not_focused, last_logged_time
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Face Recognition
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

            if name == "Unknown":
                logging.warning(f'Unknown person detected at {time.strftime("%Y-%m-%d %H:%M:%S")}')
                cv2.putText(frame, "ALERT: Unknown Person Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    process_this_frame = not process_this_frame

    # Gaze Tracking
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = gaze._face_detector(gray_frame)

    for i, face in enumerate(faces):
        gaze.refresh(frame, face)
        frame = gaze.annotated_frame()
        text = ""
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (50, 100 + 30*i), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        current_time = time.time()
        if current_time - last_logged_time >= 0.5:  
            if(text != ""):
                time_not_focused = -1
            time_not_focused += 1
            print(time_not_focused)
            if(time_not_focused >= alert_time):
                cv2.putText(frame, "Alert!", (90, 400), cv2.FONT_HERSHEY_DUPLEX, 9, (0,0,255), 15)
                logging.info(f'Person: {i} Left pupil: {left_pupil} - Right pupil: {right_pupil} - Alert: Yes')
            else:
                logging.info(f'Person: {i} Left pupil: {left_pupil} - Right pupil: {right_pupil} - Alert: No')
            last_logged_time = current_time


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    return frame

demo = gr.Interface(
    inference, 
    gr.Image(sources=["webcam"], streaming=True), 
    "image",
    live=True
)
demo.launch()
