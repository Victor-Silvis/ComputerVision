'''Main script to run model on (live)video'''

import cv2
import os
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model


#path to content
video_path = os.path.join('video', 'parking_1920_1080.mp4')
mask_path = os.path.join('mask', 'mask_1920_1080.png')

#open content
cap = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, 0)

#Preprocessing Locations based on mask
num_labels, labels = cv2.connectedComponents(mask, 4, cv2.CV_32S)
parking_spaces_list = []

for label in range(1, num_labels):  # Start from 1 to ignore the background (label 0)
    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        parking_spaces_list.append((x, y, w, h))


#Load Model
mymodel = load_model('parkingspotModel.keras')
print('Model successfully loaded')

#Display video with overlay of identified spaces
parking_results = {i: None for i in range(len(parking_spaces_list))}
diff = {i: None for i in range(len(parking_spaces_list))}

#Init variables
frame_counter = 9
previuous_frame = None
taken_spots = len(parking_spaces_list)
total_spots = len(parking_spaces_list)
spots_with_activity = [i for i in range(len(parking_spaces_list))]

#Helper function to determine activity
def calc_diff(img1, img2):
    img2 = cv2.resize(img2,(15,15))
    img2 = np.array(img2) / 255.0
    return abs(np.mean(img1) - np.mean(img2))


#Run the video and apply model on it
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("reached end of video or error")
        break

    frame_width = frame.shape[0]
    frame_height = frame.shape[1]
    frame_counter += 1

    if frame_counter %10 == 0:

        spot_imgs = []
        for idx, (x, y, w, h) in enumerate(parking_spaces_list):
            spot = frame[y:y+h, x:x+w]
            img = cv2.resize(spot,(15,15))
            img_array = np.array(img) / 255.0
            spot_img = np.expand_dims(img_array, axis=0)
            spot_imgs.append(spot_img)

            if previuous_frame is not None:
                difference = calc_diff(img_array, previuous_frame[y:y+h, x:x+w])
                if difference > 0.015:
                    spots_with_activity.append(idx)

        if spot_imgs:

            #Determine the images that the model should check (effeciency)
            imgs_to_check = [spot_imgs[i] for i in spots_with_activity]

            if imgs_to_check:
                batch_spot_imgs = np.vstack(imgs_to_check)

                #Run model on parkingspots with activity (diff in pixels)
                results = mymodel.predict(batch_spot_imgs)

                for idx, spot in enumerate(spots_with_activity):
                    result = results[idx]
                    parking_results[spot] = result

                if parking_results:
                    taken_spots = sum(1 for value in parking_results.values() if value < 0.5)
                    total_spots = len(parking_results)

        previuous_frame = frame
        
    # Section to draw the results, Draw rectangles based on the stored results
    for idx, (x, y, w, h) in enumerate(parking_spaces_list):

        #Draw the green or red boxes based on model output
        if parking_results[idx] is not None:
            if parking_results[idx] < 0.5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    #Draw the count of spots taken top-left of screen
    text = str(taken_spots) + ' / ' + str(total_spots) + ' spots available'
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    font_scale = 0.75
    thickness = 2
    position = (30,40)   
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    baseline += thickness
    rect_top_left = (position[0] - 5, position[1] - text_height - 5)
    rect_bottom_right = (position[0] + text_width + 5, position[1] + baseline)
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


    spots_with_activity = []
    frame = cv2.resize(frame, (1920,1080))
    cv2.imshow('Video Frame with boxes', frame)

    #wait for 15 ms before displaying next frame, press q to exit earlier
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

#Release video object and close any window
cap.release()
cv2.destroyAllWindows()



