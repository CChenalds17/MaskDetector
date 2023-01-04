from tensorflow.keras.models import load_model
import cv2
import numpy as np
import serial
import face_recognition

import time

# Connect arduino
# ser = serial.Serial('COM7', 9600, timeout=1)

# load mask detector model
model = load_model('saved_models/MobileNetModel.h5')
size = (224, 224)

# import face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# face detecting function
def detect(frame):
    roi_color = []
    dst = np.zeros(shape=(224,224))
    f = cv2.normalize(frame, dst, 0, 255, cv2.NORM_L1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face_locations = face_recognition.face_locations(gray)
    face_locations = face_cascade.detectMultiScale(gray, 1.3, 5)
    out_rois = []
    top = 0
    right = 0
    bottom = 0
    left = 0
    for face_location in face_locations:
        # cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        print(face_location)
        right, bottom, width, height = face_location
        left = right + width
        top = bottom + height
        roi_color = frame[bottom:top, left:right]
        out_rois.append(roi_color)
    print('out_rois:')
    print(out_rois)
    return frame, out_rois, top, right, bottom, left

# ser.write(b'L')
prev_unmasked_flag = False
curr_unmasked_flag = False
# initialize model
test_pred = model.predict(np.random.random([1, size[0], size[1], 3]))
# start video capture
video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
    # captures video_capture frame by frame
    ret, frame = video_capture.read()
    num_people = 0
    
    # gets list of faces and positions  
    canvas, faces, top, right, bottom, left = detect(frame)

    if len(faces) > 0:
        # for each face:
        for face in faces:
            temp_roi = np.array(face)
            print(temp_roi)
            roi = cv2.resize(temp_roi, size, interpolation = cv2.INTER_AREA)
            roi = (roi.astype('float')/255.0)
            roi = np.reshape(roi,[1, size[0], size[1], 3])
            # if there ARE faces:
            if np.sum([roi])!=0:
                pred = model.predict([[roi]])
                pred = pred[0] #hack hack hack code hacking
                unmasked_confidence = pred[0]
                masked_confidence = pred[1]
                # if unmasked:
                if unmasked_confidence >= masked_confidence:
                    label = 'NO MASK: {:4.2f}%'.format(unmasked_confidence * 100)
                    color = (0,0,255)
                    num_people = num_people + 1
                    key_frame = frame
                    curr_unmasked_flag = True
                # if masked:
                else:
                    label = 'MASK: {:4.2f}%'.format(masked_confidence * 100)
                    color = (0,255,0)
                    curr_unmasked_flag = False

                # label stuff onscreen
                label_position = (left, top)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)

    # if there ARE NO faces
    else:
        cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        curr_unmasked_flag = False

    # label stuff onscreen
    cv2.putText(frame, "NO MASKS : " + str(num_people), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Displays the result on camera feed                    
    cv2.imshow('MASK DETECTOR', canvas)
 
    # if changed to an unmasked face
    if curr_unmasked_flag != prev_unmasked_flag and curr_unmasked_flag:
        # trigger relay to spray water
        # ser.write(b'H')
        print('spray')
    # if changed to all masked
    elif curr_unmasked_flag != prev_unmasked_flag and not curr_unmasked_flag:
        # ser.write(b'L')
        print('stop spraying')
    prev_unmasked_flag = curr_unmasked_flag

    # The control breaks once q key is pressed                       
    if cv2.waitKey(1) & 0xff == ord('q'):              
        break
 
# Release the capture once all the processing is done.
video_capture.release()                                
cv2.destroyAllWindows()
