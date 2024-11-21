# Project Description

Using the YOLOv3 methodology, a fully customizable, object-based algorithm has been developed from scratch. The model is trained to analyze aircraft turnarounds using live video footage but is can be trained so it can be applied to a wide range of purposes. This code serves as a foundation for an analytical tool designed to evaluate various aspects of the aircraft turnaround process. Potential applications include real-time monitoring and timing of tasks performed around the aircraft or quality control checks, such as verifying proper cone placement or ensuring PPE compliance. The model achieves both high accuracy and real-time performance, maintaining a processing speed of 30+ FPS.

## Contents

- Config.py  :  File to set variables
- Dataloader.py : Script to load/feed training data to model
- Loss.py : Custom loss function for model
- Model.py : Object Detection Model itself
- Train.py : Main script for training the model
- Utils.py  : Contains various helper functions
- Demo.py  : Example of how to apply model on video
