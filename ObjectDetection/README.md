# Project Description

Using the YOLOv3 methodology, a fully customizable, object-based algorithm has been developed from scratch. The model is trained to analyze aircraft turnarounds using live video footage but is can be trained so it can be applied to a wide range of purposes. This code serves as a foundation for an analytical tool designed to evaluate various aspects of the aircraft turnaround process. Potential applications include real-time monitoring and timing of tasks performed around the aircraft or quality control checks, such as verifying proper cone placement or ensuring PPE compliance. The model achieves both high accuracy and real-time performance, maintaining a processing speed of 30+ FPS.

## 📂 Repository Contents  

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
   Use the provided `Train.py` script to train the model on your custom dataset.  

3. **Apply the Model**  
   Use `Demo.py` to analyze videos with the trained model.  


## 📁 Training Data Structure  

Organize your training data in the following structure:

.
├── ...
├── test                    # Test files (alternatively `spec` or `tests`)
│   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ...
