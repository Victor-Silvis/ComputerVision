'''Script to demo the model (Image or Video)'''

import config
from utils import ApplyModelOnContent
from model import Model
import cv2

#Load model and Script for applying (trained) model on content
model = Model(num_classes=config.NUM_CLASSES).to(config.DEVICE)
applymodel = ApplyModelOnContent(model)

#Load video and process frames
def main(video_source=0):
    
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Process the frame and apply the model
        processed_frame = applymodel.process_frame(frame, 0.6, 0.1)

        #Display the resulting frame
        cv2.imshow('DemoVideo', processed_frame)

        #Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(video_source='testvideo.mp4', model=model)
    