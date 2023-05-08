from tensorflow.keras.models import load_model
import cv2
import numpy as np
import serial
import time

# Connect to Arduino
ser = serial.Serial('COM3', 9600, timeout=1)

# Load mask detector model
model = load_model('saved_models/MobileNetModel.h5')
SIZE = (224, 224)

# Import face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face detection function
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_cascade.detectMultiScale(gray, 1.3, 5)
    out_locations = []
    x_pos, y_pos, width, height = 0, 0, 0, 0
    for face in face_locations:
        x_pos, y_pos, width, height = face
        out_locations.append((x_pos, y_pos, width, height))
    return out_locations

# Make sure sprayer is off
ser.write(b'L')
prev_unmasked_flag = False
curr_unmasked_flag = False

# Initialize model
test_pred = model.predict(np.random.random([1, SIZE[0], SIZE[1], 3]))

# Start video capture
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():

    # Capturues video_capture frame by frame
    ret, frame = video_capture.read()
    if not ret:
        break

    num_people = 0
    # Get list of face positions [(x, y, w, h), ...]
    faces = detect(frame)

    # Initialize x, y, w, h
    x, y, w, h = 0, 0, 0, 0
    # If faces ARE detected:
    if len(faces) > 0:
        # For each face:
        for face in faces:
            x, y, w, h = face
            roi = frame[y:y+h, x:x+w]
            data = cv2.resize(roi, SIZE, interpolation=cv2.INTER_AREA)
            data = data.astype('float')/255.0
            data = np.reshape(data, [1, SIZE[0], SIZE[1], 3])
            pred = model.predict(data) # [unmasked, masked]
            masked_confidence = pred[0][1] # pred[0][0]
            unmasked_confidence = pred[0][0] # 1 - masked_confidence
            # If unmasked:
            if unmasked_confidence > masked_confidence:
                label = 'UNMASKED: {:.2%}'.format(unmasked_confidence)
                color = (0, 0, 255) # (B, G, R)
                curr_unmasked_flag = True
            # If masked
            else:
                label = 'MASKED: {:.2%}'.format(masked_confidence)
                color = (0, 255, 0)
                curr_unmasked_flag = False
            
            # Draw a rectangle around each face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Label confidence percentage
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # If faces AREN'T detected
    else:
        # Label 'No Face Found'
        cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        curr_unmasked_flag = False

    # Display the camera feed
    cv2.imshow('MASK DETECTOR', frame)

    # If changed to an unmasked face
    if curr_unmasked_flag != prev_unmasked_flag and curr_unmasked_flag:
        # Trigger relay to spray water
        ser.write(b'H')
        print('SPRAY')
    # If changed to all masked
    elif curr_unmasked_flag != prev_unmasked_flag and not curr_unmasked_flag:
        # Turn off relay
        ser.write(b'L')
        print('STOP SPRAYING')
    prev_unmasked_flag = curr_unmasked_flag

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture once all the processing is done
video_capture.release()
cv2.destroyAllWindows()
