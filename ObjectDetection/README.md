# Project Description

![image7](https://github.com/user-attachments/assets/72fdce1b-5cd9-444b-96a5-94ad9b1b6e1d)

Using the YOLOv3 methodology, a fully customizable, object-based algorithm has been developed from scratch. The model is trained to analyze aircraft turnarounds using live video footage but is can be trained so it can be applied to a wide range of purposes. This code serves as a foundation for an analytical tool designed to evaluate various aspects of the aircraft turnaround process. Potential applications include real-time monitoring and timing of tasks performed around the aircraft or quality control checks, such as verifying proper cone placement or ensuring PPE compliance. The model achieves both high accuracy and real-time performance, maintaining a processing speed of 30+ FPS.

## 📂 Project Contents  

- **`Config.py`**: File to set and manage variables and configurations.  
- **`Dataloader.py`**: Script to load and feed training data to the model.  
- **`Loss.py`**: Custom loss function for the model.  
- **`Model.py`**: The YOLOv3-based object detection model.  
- **`Train.py`**: Main script to train the model.  
- **`Utils.py`**: Various helper functions to support different operations.  
- **`Demo.py`**: Example script to demonstrate the model on video input.  

---

## 🚀 How to Use  

1. **Prepare Training Data**  
   Structure your data as shown below (see *Training Data Structure*).  
   
2. **Train the Model**  
   Use the provided `Train.py` script to train the model on your custom dataset. This will save a trained model when done or manually stopped. Note for a model like this (with many layers) many epochs are needed. The training will show the accuracy on the test images every x epochs. 

3. **Apply the Model**  
   Load the saved model file and use it where you would like. Use `Demo.py` to see how to use the model on a video.

4. **Analyse**
   The model outputs all the information you need as a foundation to analyse the video. (not included in public repo)


## 📁 Training Data Structure  

Organize your training data in the following structure. Names of folders can be set in the config.py file.

```plaintext
traindata/
│
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
│
├── test/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    ├── labels/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

