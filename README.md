# 24-solvoid


# Indian Sign Language Detector
# Team Name: SOLVOID
### Team Members:
    1. Vineet Kumar Kankerwal - 
        Role: Created Data for training.
    2. Tanisha Kriplani - 
        Role: Trained Model 
    3. Yashasvi -
        Role: Prediction of Data
    4. Priyam Gupta - 
        Role: Collect samples of data and documentation.

### Tech Used:
    1.Neural Network Framework: TensorFlow Keras API
    2.Optimization Algorithm: Adam (Adaptive Moment Estimation)
    3.Image Processing Library: OpenCV (CV2)
    4.Hand Detection Library: HandTrackingModule (Python)

### Project Description

The project focuses on developing a sign language translator that detects Indian Sign Language alphabets and translates them into English alphabets. The system employs a Convolutional Neural Network (CNN) with Adam optimization for efficient learning. Utilizing the TensorFlow Keras API, the model is trained for image classification, with real-time hand detection achieved through OpenCV and the HandTrackingModule Python library. The core idea is to bridge communication gaps between differently-abled individuals and others, while also promoting awareness of Indian Sign Language.

### How to run the Project
    1. Dependencies Installation:
        Ensure you have Python installed on your system.
        Install required libraries using:
       
        pip install tensorflow opencv-Python
   
    2. Clone the Repository:
        Clone this machine to your local machine

        git clone [repository_url]

    3. Navigate to Project Directory:
        Change Directory to project folder

        cd [Project_Folder]

    4. Run the Application:
        Execute the main script to run the sign language translator:

        python prediction.py

### Explaining the Core Code
Our core code captures video from the webcam, detects hands using the HandTrackingModule, and extracts the hand region. It resizes the hand region to a fixed size, feeds it into a sign language classifier ("trainedModel.h5"), and displays the predicted letter on the video feed. The program continuously runs in a loop, updating the predictions as the hands gesture different letters.
    
## NOTE
we are unable to upload data samples because there is too huge data that it is giving error while uploading. data more than 2.5GB

