import cv2
import logging
import time
from utils import GazeTracking

logging.basicConfig(filename='./core/logging/pupil.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

alert_time = 3
time_not_focused = 0
last_logged_time = time.time()

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()

    gaze.refresh(frame)

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

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    current_time = time.time()
    if current_time - last_logged_time >= 1:  
        if(text != ""):
            time_not_focused = -1
        time_not_focused += 1
        if(time_not_focused >= alert_time):
            cv2.putText(frame, "Alert!", (90, 400), cv2.FONT_HERSHEY_DUPLEX, 9, (0,0,255), 15)
            logging.info(f'Left pupil: {left_pupil} - Right pupil: {right_pupil} - Alert: Yes')
        else:
            logging.info(f'Left pupil: {left_pupil} - Right pupil: {right_pupil} - Alert: No')
        last_logged_time = current_time

    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
