/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 2nd March ,2024
  Purpose: Created both file (offline) and live (droidcam stream) for processing.Different task numbers entered in the command 
  prompt help us in implenting various tasks in pipeline functions present in functions.cpp
  camera Index helps us to capture video from droid camera
  
  Time travel days used : None , Submitting without extensions
*/

#include "functions.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

// Task 12: Apply blur, HSV darkening, and dynamic thresholding, then display and save the results.
// Task 21: Perform the same steps as Task 12, then apply erosion, display, and save the cleaned video.
// Task 31: After preprocessing (blur, HSV darkening, dynamic thresholding, erosion), perform region growing segmentation.
// Task 32: Implement graph cut segmentation on preprocessed video frames and display the result.
// Task 33: Use connected components for segmentation on preprocessed frames to analyze segmentation techniques.
// Task 41: Detect contours on segmented frames, draw bounding boxes, and save the video output.
// Task 51: After segmentation and contour detection, enter training mode to save features to a CSV file upon a specific key press.
// Task 61: Utilize saved features for testing, classify using nearest neighbor, display classification on video stream.
// Task 71: Like Task 61, but with an added step to generate and display a confusion matrix for classification accuracy.
// Task 81: Implement K-Nearest Neighbors classification on segmented and contour-detected frames, display results on the video.
// Task 91: Similar to Task 81, but with a refined approach to KNN classification

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [video_path/task_number] [task_number/camera_index]" << std::endl;
        std::cerr << "mode: file for video file processing or live for live feed processing" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string className; // Used if in training mode

    if (mode == "file") {
        if (argc != 4) {
            std::cerr << "Usage for file mode: " << argv[0] << " file <video_path> <task_number>" << std::endl;
            return -1;
        }
        std::string videoPath = argv[2];
        int taskNumber = std::stoi(argv[3]);

        // Check if task number is 51 (training mode)
        if (taskNumber == 51) {
            std::cout << "Enter a label for the object: ";
            std::getline(std::cin, className); // Capture the class name from user input
        }

        pipeline(videoPath, taskNumber, className); // Use pipeline function for video files, pass className if in training mode
    } else if (mode == "live") {
        if (argc != 4) {
            std::cerr << "Usage for live mode: " << argv[0] << " live <task_number> <camera_index>" << std::endl;
            return -1;
        }
        int taskNumber = std::stoi(argv[2]);
        int cameraIndex = std::stoi(argv[3]); // Accepting camera index from command line

        // Check if task number is 51 (training mode)
        if (taskNumber == 51) {
            
            std::cout << "Enter a label for the object: ";
            std::getline(std::cin, className); // Capture the class name from user input
        }
        //replaceFeaturesFile("features.csv");

        pipelineLiveFeed(taskNumber, cameraIndex, className); // Passing camera index to live feed function, along with className if in training mode
    } else {
        std::cerr << "Invalid mode. Use 'file' for video file processing or 'live' for live feed processing." << std::endl;
        return -1;
    }

    cv::waitKey(0); // Wait for a key press before closing, this line might need to be adjusted based on your actual application flow
    return 0;
}
