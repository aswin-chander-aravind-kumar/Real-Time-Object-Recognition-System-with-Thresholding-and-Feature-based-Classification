# Real-Time-Object-Recognition-System-with-Thresholding-and-Feature-based-Classification

This C++ project encompasses a real-time object recognition system, focusing on preprocessing, segmentation, feature extraction, and classification within video streams. Its tasks encompass a comprehensive pipeline, from initial preprocessing steps such as blur and thresholding, through segmentation techniques like region growing and graph cut, to final classification using methods like nearest neighbor and K-Nearest Neighbors. This system aims to accurately identify and classify objects in real-time video streams, facilitating applications in fields such as computer vision and automation.

## Tasks Implemented

### Task 12
Apply blur, HSV darkening, and dynamic thresholding, then display and save the results.

### Task 21
Perform the same steps as Task 12, then apply erosion, display, and save the cleaned video.

### Task 31
After preprocessing (blur, HSV darkening, dynamic thresholding, erosion), perform region growing segmentation.

### Task 32
Implement graph cut segmentation on preprocessed video frames and display the result.

### Task 33
Use connected components for segmentation on preprocessed frames to analyze segmentation techniques.

### Task 41
Detect contours on segmented frames, draw bounding boxes, and save the video output.

### Task 51
After segmentation and contour detection, enter training mode to save features to a CSV file upon a specific key press.

### Task 61
Utilize saved features for testing, classify using nearest neighbor, display classification on video stream.

### Task 71
Like Task 61, but with an added step to generate and display a confusion matrix for classification accuracy.

### Task 81
Implement K-Nearest Neighbors classification on segmented and contour-detected frames, display results on the video.

### Task 91
Similar to Task 81, but with a refined approach to KNN classification.

## Build Instructions

1. Clone the repository.
2. Install OpenCV if not already installed.
3. Build the project using c make.
4. Run the executable file.

## Usage

- Follow the instructions provided in each task to execute specific functionalities.
- Ensure that the system has access to a video stream or pre-recorded video files for processing.

## Skills Obtained

- C++ Programming
- Computer Vision
- OpenCV
- Object Recognition
- Image Processing
- Segmentation
- Feature Extraction
- Classification
- Algorithm Implementation
- Preprocessing Techniques
- Real-Time Systems
- Pattern Recognition
- Model Training
- Video Processing
- Machine Learning


