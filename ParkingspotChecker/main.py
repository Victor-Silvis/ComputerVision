import cv2
import numpy as np
from PIL import Image
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#path to content
video_path = 'video/parking_1920_1080.mp4'
mask_path = 'mask/mask_1920_1080.png'

#open content
cap = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, 0)


'''---- Get location of parking spots using the masks ----'''
#identify the parking locations based on two color mask
num_labels, labels = cv2.connectedComponents(mask, 4, cv2.CV_32S)

#variable to store locations (coordinates) of identified spaces
parking_spaces_list = []

# Analyze the label matrix and find bounding boxes for each parking spot
for label in range(1, num_labels):  # Start from 1 to ignore the background (label 0)
    # Create a binary mask where this component is white and others are black
    component_mask = (labels == label).astype(np.uint8) * 255
    
    # Find contours of the connected component (parking spot)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box for each contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Store the parking space as a tuple (x, y, width, height)
        parking_spaces_list.append((x, y, w, h))


'''---- Load Model ----'''
mymodel = load_model('parkingspotModel.keras')
print('Model successfully loaded')

'''---- Dispay video with overlay of identified parking spaces ----'''
parking_results = {i: None for i in range(len(parking_spaces_list))}
diff = {i: None for i in range(len(parking_spaces_list))}

'''---- Some variables inits ----'''
frame_counter = 9
previuous_frame = None
taken_spots = len(parking_spaces_list)
total_spots = len(parking_spaces_list)
spots_with_activity = [i for i in range(len(parking_spaces_list))]

'''---- Helper Functions ----'''
def calc_diff(img1, img2):
    img2 = cv2.resize(img2,(15,15))
    img2 = np.array(img2) / 255.0
    return abs(np.mean(img1) - np.mean(img2))


'''---- Run the video ----'''
while cap.isOpened():
    ret, frame = cap.read() #read frame from video

    #if the frame was read correctly ret is true
    if not ret:
        print("reached end of video or error")
        break

    #get dimensions of the frame
    frame_width = frame.shape[0]
    frame_height = frame.shape[1]

    #increase frame counter
    frame_counter += 1

    #every x frames, run analysis
    if frame_counter %10 == 0:

        #variable to store the imgs of all spots
        spot_imgs = []

        #crop each parking spot & and prepare img for model
        for idx, (x, y, w, h) in enumerate(parking_spaces_list):
            spot = frame[y:y+h, x:x+w]
            img = cv2.resize(spot,(15,15))
            img_array = np.array(img) / 255.0
            spot_img = np.expand_dims(img_array, axis=0)
            spot_imgs.append(spot_img)

            #Simpel check for activity at spot & and store the spots the model should check
            if previuous_frame is not None:
                difference = calc_diff(img_array, previuous_frame[y:y+h, x:x+w])
                if difference > 0.015:
                    spots_with_activity.append(idx)

        #Section that checks if spot is free or not with model
        if spot_imgs:

            #Determine the images that the model should check (effeciency)
            imgs_to_check = [spot_imgs[i] for i in spots_with_activity]

            #From the images to the check, compile them into a batch 
            if imgs_to_check:
                batch_spot_imgs = np.vstack(imgs_to_check)

                # Run the model on the entire batch
                results = mymodel.predict(batch_spot_imgs)

                # Store the output of the model into a dictonary
                for idx, spot in enumerate(spots_with_activity):
                    result = results[idx]
                    parking_results[spot] = result

                #count how many spots taken from the amount available
                if parking_results:
                    taken_spots = sum(1 for value in parking_results.values() if value < 0.5)
                    total_spots = len(parking_results)


        #Save current frame to previous frame
        previuous_frame = frame
        
    # Section to draw the results, Draw rectangles based on the stored results
    for idx, (x, y, w, h) in enumerate(parking_spaces_list):

        #Draw the green or red boxes based on model output
        if parking_results[idx] is not None:  # Only draw if there is a prediction
            if parking_results[idx] < 0.5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for empty
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for filled

    #Draw the count of spots taken top-left of screen
    text = str(taken_spots) + ' / ' + str(total_spots) + ' spots available'
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)  # White text
    font_scale = 0.75
    thickness = 2
    position = (30,40)  # X, Y coordinates (start from bottom-left)   
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Get the text size for the background, and draw background for easier readability of text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    baseline += thickness
    rect_top_left = (position[0] - 5, position[1] - text_height - 5)
    rect_bottom_right = (position[0] + text_width + 5, position[1] + baseline)
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 0), cv2.FILLED)

    # Now draw the text over the rectangle
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    #clear lists
    spots_with_activity = []

    #Resize and Display the frame in a video
    frame = cv2.resize(frame, (1920,1080))
    cv2.imshow('Video Frame with boxes', frame)

    #wait for 15 ms before displaying next frame, press q to exit earlier
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

#Release video object and close any window
cap.release()
cv2.destroyAllWindows()



