/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 2nd March ,2024
  Purpose: This helps us to use the defined functions in different cpp files. We are defining various functions 
  in detect objects in real time and also added additional supporting functions for enhancing the object detection whilst testing 
  different implentations of the same functionality 

  Time travel days used : None , Submitting without extensions
*/


#ifndef FUNCTIONS_H
#define FUNCTIONS_H


#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <vector>


using namespace cv;
using namespace std;

// Function to process video for specific tasks
void pipeline(const std::string& videoPath, int taskNumber, const std::string& className);

// Function to apply a specific blur filter to an image
int blur5x5_2(cv::Mat& src, cv::Mat& dst);

// Function to modify the HSV values of an image to darken strongly colored pixels
void applyhsvvalue(cv::Mat& src, cv::Mat& dst);

// Function to save a sequence of frames as a video
void saveVideo(const std::vector<cv::Mat>& frames, const std::string& outputPath, int frameWidth, int frameHeight, double fps);

// Function to create a directory for storing results if it does not exist
void resultsdirectory(const std::string& path);

// Function to dynamically threshold an image based on K-means clustering
void dynamicThreshold(cv::Mat& src, cv::Mat& dst, int K);

// Function to convert an OpenCV Mat object to a vector of Vec3b
void convertToVec3bVector(const cv::Mat& src, std::vector<cv::Vec3b>& output);

// Function to calculate a dynamic threshold value based on the means of two dominant colors
int calculateDynamicThreshold(const std::vector<cv::Vec3b>& means);

// Function to process live video feed for specific tasks
void pipelineLiveFeed(int taskNumber, int cameraIndex, const std::string& className);

// Function to dilate an image
void cleanDilate(cv::Mat& src, cv::Mat& dst, int dilation_size);

// Function to erode an image
void cleanErode(cv::Mat& src, cv::Mat& dst, int erosion_size);

// Function to cleanup and save processed video
void cleanupAndSaveVideo(const std::string &inputPath, const std::string &outputPath, int frameWidth, int frameHeight, double fps);

// Function to perform region growing segmentation on an image
void regionGrowing(cv::Mat& src, cv::Mat& output, std::vector<std::vector<int>>& regionIDMap, int& regionCounter);

// Function to display segmented regions with unique colors
void displaySegmentedRegions(cv::Mat& src, const std::vector<std::vector<int>>& regionIDMap, int regionCounter, int minRegionSize);

// Function to apply the GrabCut algorithm to segment the foreground from the background
void applyGrabCut(const cv::Mat &src, cv::Mat &dst, const cv::Rect &rectangle);

// Function to perform segmentation using connected components with statistics
cv::Mat segmentConnectedComponents(const cv::Mat &src);

// Function to compute and display features of segmented regions
vector<vector<float>> computeAndDisplayFeatures(Mat& image, const vector<vector<Point>>& regions);

// Function to draw oriented bounding boxes around segmented regions
void drawOrientedBoundingBox(Mat& image, const RotatedRect& box, const Point2f& centroid, double angle);

// Function to convert region ID map to contours for visualization
std::vector<std::vector<cv::Point>> convertIDMapToContours(const cv::Mat& src);

// Function to save computed features to a CSV file
void saveFeaturesToCSV(const std::string& filename, const std::vector<std::vector<float>>& features, const std::string& className, const std::vector<std::string>& featureNames);

// Function to replace the features CSV file with a new one
void replaceFeaturesFile(const std::string& filename);

#endif // FUNCTIONS_H
