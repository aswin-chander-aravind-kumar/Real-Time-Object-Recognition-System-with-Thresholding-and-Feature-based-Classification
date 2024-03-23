/*
  Shruti Pasumarti
  Aswin Chander Aravind Kumar
  Spring 2024
  CS 5330 Computer Vision

  Date: 2nd March ,2024
  Purpose: This helps us to use the defined functions in different cpp files. We are defining various functions 
  in order to do real time object detection and exploring various techniques in order to properly recognize 
  and differentiate objects in the end
  
  Time travel days used : None , Submitting without extensions
*/


#include "functions.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <stack>
#include <filesystem>
#include <cstdlib> // For std::rand and std::srand
#include <ctime>   // For std::time
#include "kmeans.h" 
#include <numeric> 
#include <climits>
#include "csv_util.h"
#include <cmath>
#include <fstream>



using namespace cv;
using namespace std;


//Code for processing the Video Outputs Obtained
void processAndSaveVideo(const std::string &videoPath, const std::string &outputPath, int frameWidth, int frameHeight, double fps, int taskNumber) {
    cv::VideoCapture cap(videoPath);
    cv::VideoWriter videoWriter(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frameWidth, frameHeight), true);

    if (!cap.isOpened() || !videoWriter.isOpened()) {
        std::cerr << "Error opening video file or video writer." << std::endl;
        return;
    }
}

// The Results directory is present in the Debug folder . This is where all the video outputs are stored
//Additionally we also stored any screenshots we took during the process
void resultsdirectory(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
}


//Implemented the blur filter from program
//We implemented the blur filter B from task 1 . Since the execution time for implementing this filter was less as 
//seen in task1 we used 2 1x5 separable filters , vertical and horizontal
int blur5x5_2(cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    // Kernel and its normalization precomputed
    float kernel[5] = {1.0f/10, 2.0f/10, 4.0f/10, 2.0f/10, 1.0f/10};

    // Vertical pass
    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            for (int k = 0; k < src.channels(); ++k) {
                float sum = 0.0f;
                for (int u = -2; u <= 2; ++u) {
                    sum += src.at<cv::Vec3b>(i + u, j)[k] * kernel[u + 2];
                }
                dst.at<cv::Vec3b>(i, j)[k] = static_cast<uchar>(sum);
            }
        }
    }

    cv::Mat temp = dst.clone();

    // Horizontal pass
    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            for (int k = 0; k < src.channels(); ++k) {
                float sum = 0.0f;
                for (int v = -2; v <= 2; ++v) {
                    sum += temp.at<cv::Vec3b>(i, j + v)[k] * kernel[v + 2];
                }
                dst.at<cv::Vec3b>(i, j)[k] = static_cast<uchar>(sum);
            }
        }
    }

    return 0;  // Return 0 to indicate successful execution
}

//As suggested by professor for the preprocessing used applyhsvvalue function 
//making strongly colored pixels (pixels with a high saturation value)
// be darker, moving them further away from the white background (which is unsaturated)

void applyhsvvalue(cv::Mat& src, cv::Mat& dst) {
    cv::Mat hsv;
    // Convert to HSV color space
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    for (int i = 0; i < hsv.rows; ++i) {
        for (int j = 0; j < hsv.cols; ++j) {
            cv::Vec3b& pixel = hsv.at<cv::Vec3b>(i, j);
            // Check if the pixel is strongly colored (high saturation)
            if (pixel[1] > 150) { // Assuming saturation > 150 as 'strongly colored'
                // Reduce the value/brightness to make the pixel darker
                pixel[2] = std::max(0, pixel[2] - 50);
            }
        }
    }

    // Convert back to BGR from HSV
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}


// Code for saving the video
void saveVideo(const std::vector<cv::Mat>& frames, const std::string& outputPath, int frameWidth, int frameHeight, double fps) {
    cv::VideoWriter videoWriter(outputPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frameWidth, frameHeight), true);

    if (!videoWriter.isOpened()) {
        std::cerr << "Could not open the output video for write: " << outputPath << std::endl;
        return;
    }

    for (const auto& frame : frames) {
        videoWriter.write(frame);
    }

    videoWriter.release();
}



// Function to convert cv::Mat to vector<cv::Vec3b>
void convertToVec3bVector(const cv::Mat& src, std::vector<cv::Vec3b>& output) {
    output.clear();
    if (src.channels() == 3) {
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                output.push_back(src.at<cv::Vec3b>(i, j));
            }
        }
    }
}


// Calculates the dynamic threshold based on K-means clustering
int calculateDynamicThreshold(const std::vector<cv::Vec3b>& means) {
    //there are exactly 2 means
    if (means.size() != 2) {
        std::cerr << "K-means clustering did not result in 2 clusters." << std::endl;
        return 128; // Default threshold
    }

    int mean1 = std::accumulate(means[0].val, means[0].val + 3, 0) / 3;
    int mean2 = std::accumulate(means[1].val, means[1].val + 3, 0) / 3;

    return (mean1 + mean2) / 2; // Average of the two means
}

//Running a k-means algorithm on a random sample of pixel values (e.g. using 1/16 of the pixels in the image)
// with K=2 (also called the ISODATA algorithm) gives you the means of the two dominant colors in the image.
// Applies dynamic thresholding using K-means clustering

void dynamicThreshold(cv::Mat& src, cv::Mat& dst, int K = 2, int maxIterations = 10, double epsilon = 1.0, float downsamplingFactor = 0.25) {
    // Downsample the image
    cv::Mat downsampled;
    cv::resize(src, downsampled, cv::Size(), downsamplingFactor, downsamplingFactor, cv::INTER_LINEAR);

    // Convert downsampled image to a vector of Vec3b for processing
    std::vector<cv::Vec3b> data;
    if (downsampled.channels() == 3) {
        data.assign(downsampled.begin<cv::Vec3b>(), downsampled.end<cv::Vec3b>());
    } else {
        cv::Mat converted;
        cv::cvtColor(downsampled, converted, cv::COLOR_GRAY2BGR);
        data.assign(converted.begin<cv::Vec3b>(), converted.end<cv::Vec3b>());
    }

    std::vector<cv::Vec3b> means;
    std::vector<int> labels;

    // Perform K-means clustering 
    kmeans(data, means, labels, K, maxIterations, epsilon);

    // Calculate the dynamic threshold based on the clustering result
    int thresholdValue = calculateDynamicThreshold(means);

    // Apply the threshold to the original image
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    cv::threshold(dst, dst, thresholdValue, 255, cv::THRESH_BINARY);
}


// Cleaning up the binary Image
// erosion function -Eliminating thin links / reducing noise
void cleanErode(cv::Mat& src, cv::Mat& dst, int erosion_size) {
    dst = src.clone();
    for (int i = erosion_size; i < src.rows - erosion_size; ++i) {
        for (int j = erosion_size; j < src.cols - erosion_size; ++j) {
            uchar min_val = 255;
            for (int x = -erosion_size; x <= erosion_size; ++x) {
                for (int y = -erosion_size; y <= erosion_size; ++y) {
                    uchar val = src.at<uchar>(i + x, j + y);
                    if (val < min_val) {
                        min_val = val;
                    }
                }
            }
            dst.at<uchar>(i, j) = min_val;
        }
    }
}

//  dilation function - closing small regions
void cleanDilate(cv::Mat& src, cv::Mat& dst, int dilation_size) {
    dst = src.clone();
    for (int i = dilation_size; i < src.rows - dilation_size; ++i) {
        for (int j = dilation_size; j < src.cols - dilation_size; ++j) {
            uchar max_val = 0;
            for (int x = -dilation_size; x <= dilation_size; ++x) {
                for (int y = -dilation_size; y <= dilation_size; ++y) {
                    uchar val = src.at<uchar>(i + x, j + y);
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            dst.at<uchar>(i, j) = max_val;
        }
    }
}

//Code for debugging the video output in order to sae the video properly
// While saving the video I had a lot of problems because it wasn't recognizing formats
// Hence I downloaded the codec package directly within my projects folder. Obtained better results but could enhance it more
// Since I have both offline and live pipelineI saved all live feed video results with the taskname_lv format
void cleanupAndSaveVideo(const std::string &inputPath, const std::string &outputPath, int frameWidth, int frameHeight, double fps) {
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening thresholded video file: " << inputPath << std::endl;
        return;
    }

    cv::VideoWriter videoCleaned(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frameWidth, frameHeight), false);
    if (!videoCleaned.isOpened()) {
        std::cerr << "Error opening video writer for cleaned video: " << outputPath << std::endl;
        return;
    }

    cv::Mat frame, cleanedFrame;
    while (true) {
        bool isSuccess = cap.read(frame);
        if (!isSuccess || frame.empty()) break;

        // Testing if doing the combination of both erosion thereafter applying dilation on the eroded frame would give us a better result
        cleanErode(frame, cleanedFrame, 1); // Apply erosion
        cleanDilate(cleanedFrame, cleanedFrame, 1); // Apply dilation
        videoCleaned.write(cleanedFrame);
    }

    std::cout << "Cleanup and save video completed for: " << outputPath << std::endl;
    cap.release();
    videoCleaned.release();
}

struct Pixel {
    int x;
    int y;
    Pixel(int _x, int _y) : x(_x), y(_y) {}
};


// For segmenting images into regions tried a mixture of algorithms to observe the results and analyze
// Implemented region growing algorithm from scratch
// Implemented the predefined Grabcut algorithm to apply graph cut algorithm
// Referred to the psedocode present in the lecture notes to write the program from scratch
void regionGrowing(cv::Mat& src, std::vector<std::vector<int>>& regionIDMap, int& regionCounter) {
    int numRows = src.rows;
    int numCols = src.cols;

    std::stack<Pixel> pixelStack;

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            if (src.at<uchar>(i, j) == 1 && regionIDMap[i][j] == 0) {
                ++regionCounter;
                regionIDMap[i][j] = regionCounter;
                pixelStack.push(Pixel(j, i));

                while (!pixelStack.empty()) {
                    Pixel currentPixel = pixelStack.top();
                    pixelStack.pop();

                    // Define neighbors
                    std::vector<Pixel> neighbors = {
                        Pixel(currentPixel.x + 1, currentPixel.y),
                        Pixel(currentPixel.x - 1, currentPixel.y),
                        Pixel(currentPixel.x, currentPixel.y + 1),
                        Pixel(currentPixel.x, currentPixel.y - 1)
                    };

                    for (const Pixel& neighbor : neighbors) {
                        // Check bounds
                        if (neighbor.x >= 0 && neighbor.x < numCols && neighbor.y >= 0 && neighbor.y < numRows) {
                            if (src.at<uchar>(neighbor.y, neighbor.x) == 1 && regionIDMap[neighbor.y][neighbor.x] == 0) {
                                regionIDMap[neighbor.y][neighbor.x] = regionCounter;
                                pixelStack.push(neighbor);
                                std::cout << "Added neighbor: (" << neighbor.x << ", " << neighbor.y << ") to region " << regionCounter << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
    //std::cout << "Region Counter: " << regionCounter << std::endl;
    //std::cout << "Region ID Map:" << std::endl;
    
}

// Chose 8 connected graph to segment the region
void displaySegmentedRegions(cv::Mat& src, std::vector<std::vector<int>>& regionIDMap, int regionCounter, int minRegionSize) {
    cv::RNG rng(12345); // Random number generator for colors

    // Calculate connected components
    cv::Mat labels, stats, centroids;
    int connectivity = 8;
    int numLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, connectivity, CV_32S);

    // Filter small regions and assign colors to remaining regions
    std::vector<cv::Vec3b> colors(numLabels);
    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minRegionSize) {
            colors[i] = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        }
    }

    // Display segmented regions
    cv::Mat segmented = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < segmented.rows; ++y) {
        for (int x = 0; x < segmented.cols; ++x) {
            int label = labels.at<int>(y, x);
            if (label > 0 && stats.at<int>(label, cv::CC_STAT_AREA) >= minRegionSize) {
                segmented.at<cv::Vec3b>(y, x) = colors[label];
            }else {
                segmented.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        }
    }
    }

    // Display the segmented image
    cv::imshow("Segmented Regions", segmented);
    
}

//Applying graph cut
void applyGrabCut(const cv::Mat &src, cv::Mat &dst, const cv::Rect &rectangle) {
    // Initialize the mask, background and foreground models
    cv::Mat mask, bgModel, fgModel;
    mask.create(src.size(), CV_8UC1);
    mask.setTo(cv::GC_BGD); // Set background
    (mask(rectangle)).setTo(cv::Scalar(cv::GC_PR_FGD)); // Set probable foreground for the region within the rectangle

    // Apply GrabCut algorithm
    cv::grabCut(src, mask, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);

    // Create a mask with the pixels marked as foreground and probable foreground
    cv::Mat foregroundMask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);

    // Generate output image
    src.copyTo(dst, foregroundMask); // Copy src image to dst where the foregroundMask is true
}

// Function to perform segmentation using 8 connected
cv::Mat segmentConnectedComponents(const cv::Mat &src) {
    cv::Mat labels, stats, centroids;
    cv::Mat binaryImage;
    cv::threshold(src, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Perform analysis on each segment
    int numLabels = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids);

    // Create an output image to draw the segmentation result
    std::vector<cv::Vec3b> colors(numLabels);
    for(int i = 1; i < numLabels; i++) {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    // Assign colors to the labels
    // When I placed objects on white background. The backround was a representation of a mixture of R,G,B colors
    // The segmented objects are dark. So when we place a white object on white program it isnt segmented nor it is
    //visible in thresholding
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; r++) {
        for(int c = 0; c < dst.cols; c++) {
            int label = labels.at<int>(r, c);
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
            pixel = colors[label];
        }
    }
    return dst;
}

//Drawing bounding box
// In some cases the bounding box is present on the whole frame . In an instance when the camera is placed very far
// When it is close to the objects the outlines of the segments were detected
void drawOrientedBoundingBox(Mat& image, const RotatedRect& box, const Point2f& centroid, double angle) {
    Point2f vertices[4];
    box.points(vertices);
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

    // Compute and draw the axis of the least moment
    Point2f endPoint(centroid.x + 100 * cos(angle), centroid.y + 100 * sin(angle));
    line(image, centroid, endPoint, Scalar(0, 0, 255), 2);
}

// Computing and Displaying Features
vector<vector<float>> computeAndDisplayFeatures(Mat& image, const vector<vector<Point>>& regions) {
    vector<vector<float>> allFeatures;

    for (size_t i = 0; i < regions.size(); ++i) {
        vector<float> features;
        Moments mu = moments(regions[i]);

        // Calculate centroid
        Point2f centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);

        // Calculate oriented bounding box
        RotatedRect box = minAreaRect(regions[i]);

        // Calculate angle of the axis of the least moment
        double angle = 0.5 * atan2(2 * mu.mu11, mu.mu20 - mu.mu02);

        // Aspect Ratio and Percent Filled
        double aspectRatio = box.size.width / box.size.height;
        double area = contourArea(regions[i]);
        double filledArea = mu.m00;
        double percentFilled = 0;
        if (area > 0) { // Check to avoid division by zero
            percentFilled = (filledArea / area) * 100;
        }

        // Add computed features to the vector
        features.push_back(i + 1);  // Region ID
        features.push_back(centroid.x); // Centroid X
        features.push_back(centroid.y); // Centroid Y
        features.push_back(static_cast<float>(angle)); // Orientation in radians, converted to degrees
        features.push_back(static_cast<float>(percentFilled)); // Percent Filled
        features.push_back(static_cast<float>(aspectRatio)); // Aspect Ratio
        allFeatures.push_back(features);

        // Visualize the oriented bounding box and axis on the image
        drawOrientedBoundingBox(image, box, centroid, angle);

        // display features 
        cout << "Region " << i + 1 << ": Aspect Ratio = " << aspectRatio
             << ", Percent Filled = " << percentFilled << "%" << endl;
        cout << "Region ID: " << i + 1 << endl;
        cout << "Centroid: (" << centroid.x << ", " << centroid.y << ")" << endl;
        cout << "Orientation: " << angle * (180.0/CV_PI) << " degrees" << endl; // Convert radians to degrees
    }

    return allFeatures;
}

//Saving features to a csv file
void saveFeaturesToCSV(const std::string& filename, 
                       const std::vector<std::vector<float>>& features, 
                       const std::string& className, 
                       const std::vector<std::string>& featureNames) {
    std::ofstream file(filename, std::ios::out | std::ios::app); // Open in append mode
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Check if file is empty to decide whether to write headers
    file.seekp(0, std::ios::end); // Move to end of file
    bool isEmpty = file.tellp() == 0;
    if (isEmpty) {
        // Write the headers
        file << "Class";
        for (const auto& name : featureNames) {
            file << "," << name;
        }
        file << "\n";
    }

    // Write data rows
    for (const auto& featureVector : features) {
        file << className; // Write the class name
        for (const auto& value : featureVector) {
            file << ",";
            if (std::isfinite(value)) { // Check if the value is finite before writing
                file << value;
            } else {
                file << "0"; // Write N/A for non-finite values
            }
        }
        file << "\n";
    }
    file.close();
}

//Computing Eucledian distance - considering standard deviation
float EuclideanDistance(const vector<float>& x1, const vector<float>& x2, const vector<float>& std_devs) {
    float distance = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        distance += pow((x1[i] - x2[i]) / std_devs[i], 2);
    }
    return sqrt(distance);
}

// Function to classify a new feature vector using nearest-neighbor 
string classifyNewFeature(const vector<float>& new_feature_vector, const vector<vector<float>>& known_features, const vector<string>& class_names, const vector<float>& std_devs) {
    float min_distance = numeric_limits<float>::max();
    string closest_class;
    
    for (size_t i = 0; i < known_features.size(); ++i) {
        float distance = EuclideanDistance(new_feature_vector, known_features[i], std_devs);
        if (distance < min_distance) {
            min_distance = distance;
            closest_class = class_names[i];
        }
    }
    
    return closest_class;
}

// Computes Euclidean distance between two feature vectors - mainly focused on using the k value
float computeEuclideanDistance(const vector<float>& vec1, const vector<float>& vec2) {
    float distance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        distance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return std::sqrt(distance);
}


// Classifies a given feature vector using K-Nearest Neighbors
string classifyWithKNN(const vector<float>& testFeature, const vector<vector<float>>& trainFeatures, const vector<string>& trainLabels, int k) {
    // Pair of (distance, index)
    vector<pair<float, int>> distances;

    // Compute distances from the test feature to all training features
    for (size_t i = 0; i < trainFeatures.size(); ++i) {
        float dist = computeEuclideanDistance(testFeature, trainFeatures[i]);
        distances.push_back(make_pair(dist, i));
    }

    // Sort distances
    sort(distances.begin(), distances.end());

    // Collect the labels of the k closest neighbors
    map<string, int> labelCounts;
    for (int i = 0; i < k; ++i) {
        string label = trainLabels[distances[i].second];
        labelCounts[label]++;
    }

    // Determine the most frequent label among the k neighbors
    string mostFrequentLabel = "";
    int maxCount = 0;
    for (const auto& labelCount : labelCounts) {
        if (labelCount.second > maxCount) {
            mostFrequentLabel = labelCount.first;
            maxCount = labelCount.second;
        }
    }

    return mostFrequentLabel;
}



// Assuming regionIDMap is a 2D vector<int> representing different regions by integers
cv::Mat regionIDMapToMat(const std::vector<std::vector<int>>& regionIDMap) {
    cv::Mat binaryImage(regionIDMap.size(), regionIDMap[0].size(), CV_8U, cv::Scalar(0));
    for (size_t i = 0; i < regionIDMap.size(); ++i) {
        for (size_t j = 0; j < regionIDMap[i].size(); ++j) {
            if (regionIDMap[i][j] > 0) { // Assuming non-zero values indicate a region
                binaryImage.at<uchar>(i, j) = 255;
            }
        }
    }
    return binaryImage;
}

//Converting the obtained ID map to contours
vector<vector<Point>> convertIDMapToContours(const Mat& src) {
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}



// Tried this function so that I can replace the features.csv file after certain no of iterations in order to have clean data
// Couldn't implement it on time though. We have to figure out the logic more
void replaceFeaturesFile(const std::string& filename) {
    std::filesystem::path filePath{filename};
    if (std::filesystem::exists(filePath)) { // Check if the file exists
        std::filesystem::remove(filePath); // Delete the file
    }

}


// Function to read features and class names from the CSV file
void readFeaturesAndClassNames(const string& filename, vector<vector<float>>& known_features, vector<string>& class_names) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file for reading: " << filename << endl;
        return;
    }
    
    string line, value, class_name;
    // Skip the header line
    getline(file, line);
    while (getline(file, line)) {
        istringstream iss(line);
        
        // Read the class name
        getline(iss, class_name, ',');
        class_names.push_back(class_name);

        // Read the feature values
        vector<float> feature_values;
        while (getline(iss, value, ',')) {
            feature_values.push_back(stof(value));
        }
        known_features.push_back(feature_values);
    }
    file.close();
}

// Function to calculate standard deviations for each feature
vector<float> calculateStandardDeviations(const vector<vector<float>>& known_features) {
    vector<float> std_devs(known_features[0].size(), 0); // Initialize standard deviations
    vector<float> means(known_features[0].size(), 0); // Initialize means

    // Compute means
    for (const auto& features : known_features) {
        for (size_t i = 0; i < features.size(); ++i) {
            means[i] += features[i];
        }
    }
    for (float& mean : means) {
        mean /= known_features.size();
    }

    // Compute standard deviations
    for (const auto& features : known_features) {
        for (size_t i = 0; i < features.size(); ++i) {
            std_devs[i] += pow(features[i] - means[i], 2);
        }
    }
    for (float& std_dev : std_devs) {
        std_dev = sqrt(std_dev / known_features.size());
    }
    
    return std_devs;
}


// Function to classify a feature vector using scaled Euclidean distance
string classifyFeatureVector(const vector<float>& feature_vector, const vector<vector<float>>& known_features, const vector<string>& class_names, const vector<float>& std_devs) {
    string closest_class;
    float min_distance = numeric_limits<float>::max();

    for (size_t i = 0; i < known_features.size(); ++i) {
        float distance = 0.0;
        for (size_t j = 0; j < feature_vector.size(); ++j) {
            distance += pow((feature_vector[j] - known_features[i][j]) / std_devs[j], 2);
        }
        distance = sqrt(distance);
        if (distance < min_distance) {
            min_distance = distance;
            closest_class = class_names[i];
        }
    }
    
    return closest_class;
}

// This is another implementation of K Features. Two of us implemented same algoritm
string classifyKFeatureVector(const vector<float>& feature_vector, const vector<vector<float>>& known_features, const vector<string>& class_names, const vector<float>& std_devs, int k) {
    // Store distances and corresponding class indices
    vector<pair<float, size_t>> distances_and_indices;

    // Calculate distances between feature_vector and known_features
    for (size_t i = 0; i < known_features.size(); ++i) {
        float distance = 0.0;
        for (size_t j = 0; j < feature_vector.size(); ++j) {
            distance += pow((feature_vector[j] - known_features[i][j]) / std_devs[j], 2);
        }
        distance = sqrt(distance);
        distances_and_indices.emplace_back(distance, i);
    }

    // Sort distances in ascending order
    sort(distances_and_indices.begin(), distances_and_indices.end());

    // Count occurrences of each class among the k nearest neighbors
    unordered_map<string, int> class_counts;
    for (int i = 0; i < k; ++i) {
        size_t index = distances_and_indices[i].second;
        string cls = class_names[index];
        class_counts[cls]++;
    }

    // Find the class with the highest count
    string closest_class;
    int max_count = numeric_limits<int>::min();
    for (const auto& pair : class_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            closest_class = pair.first;
        }
    }

    return closest_class;
}


// This pipeline function was implemented initially because one of us is using a ROG which doesn't have inbuilt camera
// Created this pipeline in order to invoke a video file offline and test the program initially 
//Hence the main function was created in same fashion where in when we use the word file it indicates offline processing
// live - livefeed coming from an external camera
void pipeline(const std::string& videoPath, int taskNumber, const std::string& className) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Could not open the video file: " << videoPath << std::endl;
        return;
    }

    std::string resultsDir = "results";
    resultsdirectory(resultsDir);

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter videoBlurred(resultsDir + "/blurredVideo.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frameWidth, frameHeight));
    cv::VideoWriter videoDarkened(resultsDir + "/darkenedHSVVideo.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frameWidth, frameHeight));
    cv::VideoWriter videoThresholded(resultsDir + "/ThresholdedVideo.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frameWidth, frameHeight));
    cv::VideoWriter videoCleaned(resultsDir + "/CleanedVideo.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frameWidth, frameHeight)); // For task 2

    cv::Mat frame, processedFrame, darkenedFrame, thresholdedFrame,cleanedFrame;
    int processedFrames = 0;
    while (true) {
        bool isSuccess = cap.read(frame);
        if (!isSuccess) {
            std::cout << "End of video or read error at frame: " << processedFrames << std::endl;
            break;
        }

        if (taskNumber == 1) {
            blur5x5_2(frame, processedFrame);
            videoBlurred.write(processedFrame);

            applyhsvvalue(processedFrame, darkenedFrame);
            videoDarkened.write(darkenedFrame);

            // Adjust the call to include default values for maxIterations and epsilon
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0);// K=2 for ISODATA
            if (!thresholdedFrame.empty()) {
                //cv::imshow("Darkened Video", darkenedFrame);
                cv::imshow("Thresholded Video", thresholdedFrame);
                videoThresholded.write(thresholdedFrame);
                if (cv::waitKey(30) >= 0) break;
            }
        }else if (taskNumber == 2) {
            // For Task 2, reusing thresholdedFrame from Task 1 as input
            // Assuming thresholdedFrame is the input for cleaning operations
            dynamicThreshold(frame, thresholdedFrame, 2, 10, 1.0); // Assuming we need to threshold again for demonstration
            

            // Applying erosion and dilation
            //cleanErode(thresholdedFrame, cleanedFrame, 1);
            cleanDilate(cleanedFrame, cleanedFrame, 1);

            videoCleaned.write(cleanedFrame);

            if (!cleanedFrame.empty()) {
                cv::imshow("Cleaned Video", cleanedFrame);
                if (cv::waitKey(30) >= 0) break;
            }
        }

        ++processedFrames;
    }

    std::cout << "Total processed frames: " << processedFrames << std::endl;

    // Release all resources
    videoBlurred.release();
    videoDarkened.release();
    videoThresholded.release();
    videoCleaned.release(); // Release resources for task 2
    cap.release();
    cv::destroyAllWindows();
}



static std::vector<std::string> classNames;


// Pipeline for working on live feed processing 
// Somehow couldn't connect a USB camera and hence relied on Droid Cam 
void pipelineLiveFeed(int taskNumber, int cameraIndex, const std::string& className) {
    cv::VideoCapture cap(cameraIndex);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
        return;
    }

    std::string resultsDir = "results";
    resultsdirectory(resultsDir);

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter videoBlurredlv(resultsDir + "/blurredVideo_lv.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);
    cv::VideoWriter videoSegmentedlv(resultsDir + "/segmented.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);

    cv::VideoWriter videoDarkenedlv(resultsDir + "/darkenedHSVVideo_lv.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);
    cv::VideoWriter videoThresholdedlv(resultsDir + "/ThresholdedVideo_lv.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);
    // Initialize video writer for cleaned video in live feed for task 21
    cv::VideoWriter videoCleanedlv(resultsDir + "/CleanedVideo_lv.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);
    cv::VideoWriter videoGrabcutlv(resultsDir + "/GrabcutVideo.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);
    cv::VideoWriter videoSegmented33(resultsDir + "/SegmentedVideo_task33.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);
    cv::VideoWriter videoboundingbox(resultsDir + "/Videotask41.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight), true);

    cv::Mat frame, processedFrame, darkenedFrame, thresholdedFrame, cleanedFrame,grabcutFrame,boundingFrame;

    //replaceFeaturesFile("features.csv");

    while (true) {
        bool isSuccess = cap.read(frame);
        if (!isSuccess || frame.empty()) break;

        // Display the raw video
        cv::imshow("Live - Raw Video", frame);

        // Task 12: Original processing without cleaning
        if (taskNumber == 12) {
            cout<<"New"<<endl;
            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);

            cv::imshow("Live - Blurred Video", processedFrame);
            videoBlurredlv.write(processedFrame);

            cv::imshow("Live - Darkened Video", darkenedFrame);
            videoDarkenedlv.write(darkenedFrame);

            cv::imshow("Live - Thresholded Video", thresholdedFrame);
            videoThresholdedlv.write(thresholdedFrame);
        }
        // Task 21: Processing including cleaning after thresholding
        else if (taskNumber == 21) {
            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);
            //cleanDilate(thresholdedFrame, cleanedFrame, 1);

            cv::imshow("Live - Blurred Video", processedFrame);
            //videoBlurredlv.write(processedFrame);

            cv::imshow("Live - Darkened Video", darkenedFrame);
            ////videoDarkenedlv.write(darkenedFrame);

            cv::imshow("Live - Thresholded Video", thresholdedFrame);
            videoThresholdedlv.write(thresholdedFrame);

            cv::imshow("Live - Cleaned Video", cleanedFrame);
            //videoCleanedlv.write(cleanedFrame);
        }else if (taskNumber == 31) {
            // Implemented region growing for segmentation
            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);
            //cleanDilate(thresholdedFrame, cleanedFrame, 1);
            
            // Create a new frame to save the segmented output
    cv::Mat segmentedOutput = cleanedFrame.clone();

    // Display segmented output
    std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
    int regionCounter = 0;
    regionGrowing(cleanedFrame, regionIDMap, regionCounter);

    
    // Display segmented output
    int minRegionSize = 1000; // Minimum region size threshold
    displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);
        }if (taskNumber == 32) {
            // Implemented graph cuts for segmentation
            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);


            // Ensuring the frame is in the correct format for grabCut
            cv::Mat frameForGrabCut = frame; 
            cv::Rect rectangle(50, 50, frameWidth - 100, frameHeight - 100);


            // Apply GrabCut on the frameForGrabCut
            cv::Mat mask, bgModel, fgModel; // Temporary variables for GrabCut
            mask.create(frameForGrabCut.size(), CV_8UC1);
            mask.setTo(cv::GC_BGD);
            (mask(rectangle)).setTo(cv::Scalar(cv::GC_PR_FGD));

            cv::grabCut(frameForGrabCut, mask, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);
            cv::Mat foregroundMask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
            cv::Mat grabCutResult;
            frameForGrabCut.copyTo(grabCutResult, foregroundMask); // Copy to grabCutResult where the foregroundMask is true

            cv::imshow("Live - grabcut Video", grabCutResult);
            videoCleanedlv.write(grabCutResult);

        }else if (taskNumber == 33) {
            //Used the already existing segmentedConnectedComponents for segmentation to understand 
            // the differences for various implementations of segmentation
            // Found region growing segmentation giving the best results
            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);
    // Assuming cleanedFrame is already a grayscale image after thresholding and cleaning.
    if (!cleanedFrame.empty()) {
        /// Use cleanedFrame directly if it's already grayscale.
        cv::Mat segmentedFrame = segmentConnectedComponents(cleanedFrame); 

        if (!segmentedFrame.empty()) {
            cv::imshow("Segmented Frame - Task 33", segmentedFrame);
            videoSegmented33.write(segmentedFrame);
        } else {
            std::cerr << "Segmentation failed: Segmented frame is empty." << std::endl;
        }
    } else {
        std::cerr << "Pre-segmentation error: Cleaned frame is empty." << std::endl;
    }
}else if (taskNumber == 41) {
    //Creating bounding box
    blur5x5_2(frame, processedFrame);
    applyhsvvalue(processedFrame, darkenedFrame);
    dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
    cleanErode(thresholdedFrame, cleanedFrame, 1);

    // Create a new frame to save the segmented output
    cv::Mat segmentedOutput = cleanedFrame.clone();

    // Perform region growing on the cleaned frame
    std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
    int regionCounter = 0;
    regionGrowing(cleanedFrame, regionIDMap, regionCounter);

    // Display segmented output
    int minRegionSize = 1000; // Minimum region size threshold
    displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);

    std::vector<std::vector<cv::Point>> contours;
    if (!segmentedOutput.empty()) {
        cv::Mat gray;
        if (segmentedOutput.channels() > 1) {
            cv::cvtColor(segmentedOutput, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = segmentedOutput;
        }

        // Find contours from the segmented output
        cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw contours on a copy of the original frame
        boundingFrame = frame.clone();
        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(boundingFrame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("Live - Contours", boundingFrame);
        videoboundingbox.write(boundingFrame);
    }
     else {
        std::cerr << "Contour detection error: Cleaned frame is empty." << std::endl;
    }

 }else if (taskNumber == 51) {
    // After segmenting and detecting contours
    // Training and testing the data
    // Defined functions already above for implementing the testing and training process 
    // and called those functions here
    //Click n to enter training mode. We enter the label name . Continuosuly clicking n will save features to csv
    

    blur5x5_2(frame, processedFrame);
    applyhsvvalue(processedFrame, darkenedFrame);
    dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
    cleanErode(thresholdedFrame, cleanedFrame, 1);

    // Create a new frame to save the segmented output
    cv::Mat segmentedOutput = cleanedFrame.clone();

    // Perform region growing on the cleaned frame
    std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
    int regionCounter = 0;
    regionGrowing(cleanedFrame, regionIDMap, regionCounter);

    // Display segmented output
    int minRegionSize = 1000; // Minimum region size threshold
    displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);

    std::vector<std::vector<cv::Point>> contours;
    if (!segmentedOutput.empty()) {
        cv::Mat gray;
        if (segmentedOutput.channels() > 1) {
            cv::cvtColor(segmentedOutput, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = segmentedOutput;
        }

        // Find contours from the segmented output
        cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw contours on a copy of the original frame
        boundingFrame = frame.clone();
        for (size_t i = 0; i < contours.size(); i++) {
            cv::drawContours(boundingFrame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("Live - Contours", boundingFrame);
        videoboundingbox.write(boundingFrame);
        char key = cv::waitKey(30); // Use char to capture the key press
        if (key == 'q' || key == 'Q') {
            break; // Exit the loop if 'q' is pressed
        }
        if (key == 'n' || key == 'N') {
        // Compute and display features, then save them
            vector<vector<float>> features = computeAndDisplayFeatures(boundingFrame, contours);

        // Specify the filename for the CSV file
            string csvFilename = "features.csv";

        // Define feature names correctly (missing comma in your original code)
            std::vector<std::string> featureNames = {"Region ID", "Centroid X", "Centroid Y", "Orientation", "Percent Filled", "Aspect Ratio"};


        // Tried defining the saveFeaturesToCSV file such that in the first iteration it divides
        // the csv file in different categories , className and different computedFeatures as Aspect Ration , orientation etc
        // Then we map the values of each computed Feature to the respective categories present in csv file
            saveFeaturesToCSV(csvFilename, features, className, featureNames);
            cout << "Features saved to " << csvFilename << endl;

        }
    } else {
        std::cerr << "Contour detection error: Cleaned frame is empty." << std::endl;
    }
    
}else if (taskNumber == 61) {

        // Here we can test the already saved data in the csv features file
        // Nearest neighbour classification
        cout << "Enter testing mode" << endl;

        // Load known features and class names from the database
        vector<vector<float>> known_features;
        vector<string> class_names;
        vector<float> std_devs;

        // Reading the features from the csv file
        readFeaturesAndClassNames("features.csv", known_features, class_names);
        std_devs = calculateStandardDeviations(known_features);

        while (true) {
            bool isSuccess = cap.read(frame);
            if (!isSuccess || frame.empty()) break;

            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);

            // Create a new frame to save the segmented output
            cv::Mat segmentedOutput = cleanedFrame.clone();

            // Perform region growing on the cleaned frame
            std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
            int regionCounter = 0;
            regionGrowing(cleanedFrame, regionIDMap, regionCounter);

            // Display segmented output
            int minRegionSize = 1000; // Minimum region size threshold
            displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);

            std::vector<std::vector<cv::Point>> contours;
            if (!segmentedOutput.empty()) {
                cv::Mat gray;
            if (segmentedOutput.channels() > 1) {
                cv::cvtColor(segmentedOutput, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = segmentedOutput;
            }

            // Find contours from the segmented output
            cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Draw contours on a copy of the original frame
            boundingFrame = frame.clone();
            for (size_t i = 0; i < contours.size(); i++) {
                cv::drawContours(boundingFrame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Live - Contours", boundingFrame);
            videoboundingbox.write(boundingFrame);

            // Compute features for contours
            vector<vector<float>> features = computeAndDisplayFeatures(boundingFrame, contours);

            // Classify features and visualize
            for (const auto& feature : features) {
                string classified_class = classifyFeatureVector(feature, known_features, class_names, std_devs);
                // Visualize the classified class on the video stream
                // Assume 'centroid' can be computed from the 'feature'
                Point centroid = Point(static_cast<int>(feature[1]), static_cast<int>(feature[2]));
                putText(frame, classified_class, centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            }

            // Show the frame with the classification
            cv::imshow("Live Feed - Classified Objects", frame);

            // Break the loop if the user presses 'q' or ESC
            char key = cv::waitKey(30);
            if (key == 'q' || key == 27) { // ESC key or 'q' key to quit
                break;
            }
        }
        }
}else if (taskNumber == 71) {
        // Generating and displaying a confusion matrix for classification accuracy.
        cout << "Enter testing mode" << endl;

        // Load known features and class names from the database
        vector<vector<float>> known_features;
        vector<string> class_names;
        vector<float> std_devs;

        readFeaturesAndClassNames("features.csv", known_features, class_names);
        std_devs = calculateStandardDeviations(known_features);

        while (true) {
            bool isSuccess = cap.read(frame);
            if (!isSuccess || frame.empty()) break;

            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);

            // Create a new frame to save the segmented output
            cv::Mat segmentedOutput = cleanedFrame.clone();

            // Perform region growing on the cleaned frame
            std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
            int regionCounter = 0;
            regionGrowing(cleanedFrame, regionIDMap, regionCounter);

            // Display segmented output
            int minRegionSize = 1000; // Minimum region size threshold
            displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);

            std::vector<std::vector<cv::Point>> contours;
            if (!segmentedOutput.empty()) {
                cv::Mat gray;
            if (segmentedOutput.channels() > 1) {
                cv::cvtColor(segmentedOutput, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = segmentedOutput;
            }

            // Find contours from the segmented output
            cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Draw contours on a copy of the original frame
            boundingFrame = frame.clone();
            for (size_t i = 0; i < contours.size(); i++) {
                cv::drawContours(boundingFrame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Live - Contours", boundingFrame);
            videoboundingbox.write(boundingFrame);

            // Compute features for contours
            vector<vector<float>> features = computeAndDisplayFeatures(boundingFrame, contours);
            std::map<std::string, int> labelMap = {
    {"box", 0},
    {"remote", 1},
    {"case", 2},
    {"glass", 3},
    {"airpods", 4}
};
    vector<vector<int>> confusionMat(5, vector<int>(5, 0));
// Populate map with label indices
    string actualLabel, predictedLabel;

    // Prompt user for the actual object label
    cout << "Enter the Actual Object Label: ";
    cin >> actualLabel;

    // Ensure the entered label exists in the label mapping
    if (labelMap.find(actualLabel) == labelMap.end()) {
        cout << "Invalid label entered. Please enter a valid label." << endl; // Exit with error
    }

    // Determine the index of the actual label
    int actualIndex = labelMap[actualLabel];

    // Perform classification (replace with your classification method)
    // For demonstration purposes, a placeholder is used here
    predictedLabel = "box"; // Replace with actual classification result

    // Determine the index of the predicted label
    int predictedIndex = labelMap[predictedLabel];

    // Update confusion matrix
    confusionMat[actualIndex][predictedIndex]++;

    // Print the confusion matrix elements
    cout << "Confusion Matrix elements:" << endl;
    for (const auto& row : confusionMat) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout << endl;
    }

    }
}
}else if (taskNumber == 81) {

        // KNN Classification
        cout << "Enter testing mode" << endl;

        // Load known features and class names from the database
        vector<vector<float>> known_features;
        vector<string> class_names;
        //vector<float> std_devs;
        int k=3;

        readFeaturesAndClassNames("features.csv", known_features, class_names);
        //std_devs = calculateStandardDeviations(known_features);

        while (true) {
            bool isSuccess = cap.read(frame);
            if (!isSuccess || frame.empty()) break;

            blur5x5_2(frame, processedFrame);
            //cv::imshow("Live - Blurred Video", processedFrame);

            applyhsvvalue(processedFrame, darkenedFrame);
            //cv::imshow("Live - Darkened Video", darkenedFrame);

            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            //cv::imshow("Live - Thresholded Video", thresholdedFrame);

            cleanErode(thresholdedFrame, cleanedFrame, 1);
            //
            cv::imshow("Live - Cleaned Video", cleanedFrame);

            // Create a new frame to save the segmented output
            cv::Mat segmentedOutput = cleanedFrame.clone();

            // Perform region growing on the cleaned frame
            std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
            int regionCounter = 0;
            regionGrowing(cleanedFrame, regionIDMap, regionCounter);

            // Display segmented output
            int minRegionSize = 1000; // Minimum region size threshold
            displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);

            std::vector<std::vector<cv::Point>> contours;
            if (!segmentedOutput.empty()) {
                cv::Mat gray;
            if (segmentedOutput.channels() > 1) {
                cv::cvtColor(segmentedOutput, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = segmentedOutput;
            }

            // Find contours from the segmented output
            cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Draw contours on a copy of the original frame
            boundingFrame = frame.clone();
            for (size_t i = 0; i < contours.size(); i++) {
                cv::drawContours(boundingFrame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Live - Contours", boundingFrame);
            videoboundingbox.write(boundingFrame);

            // Compute features for contours
            vector<vector<float>> features = computeAndDisplayFeatures(boundingFrame, contours);

            // Classify features and visualize
            for (const auto& feature : features) {
                string classified_class = classifyWithKNN(feature, known_features, class_names, k);
                // Visualize the classified class on the video stream
                // Assume 'centroid' can be computed from the 'feature'
                Point centroid = Point(static_cast<int>(feature[1]), static_cast<int>(feature[2]));
                putText(frame, classified_class, centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            }

            // Show the frame with the classification
            cv::imshow("Live Feed - Classified Objects", frame);

            // Break the loop if the user presses 'q' or ESC
            char key = cv::waitKey(500);
            if (key == 'q' || key == 27) { // ESC key or 'q' key to quit
                break;
            }
            }
    }    
}   
else if (taskNumber == 91) {
        cout << "Enter testing mode" << endl;

        // Load known features and class names from the database
        vector<vector<float>> known_features;
        vector<string> class_names;
        vector<float> std_devs;

        readFeaturesAndClassNames("features.csv", known_features, class_names);
        std_devs = calculateStandardDeviations(known_features);
        int k = 3;
        while (true) {
            bool isSuccess = cap.read(frame);
            if (!isSuccess || frame.empty()) break;

            blur5x5_2(frame, processedFrame);
            applyhsvvalue(processedFrame, darkenedFrame);
            dynamicThreshold(darkenedFrame, thresholdedFrame, 2, 10, 1.0, 0.25);
            cleanErode(thresholdedFrame, cleanedFrame, 1);

            // Create a new frame to save the segmented output
            cv::Mat segmentedOutput = cleanedFrame.clone();

            // Perform region growing on the cleaned frame
            std::vector<std::vector<int>> regionIDMap(thresholdedFrame.rows, std::vector<int>(thresholdedFrame.cols, 0));
            int regionCounter = 0;
            regionGrowing(cleanedFrame, regionIDMap, regionCounter);

            // Display segmented output
            int minRegionSize = 1000; // Minimum region size threshold
            displaySegmentedRegions(segmentedOutput, regionIDMap, regionCounter, minRegionSize);

            std::vector<std::vector<cv::Point>> contours;
            if (!segmentedOutput.empty()) {
                cv::Mat gray;
            if (segmentedOutput.channels() > 1) {
                cv::cvtColor(segmentedOutput, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = segmentedOutput;
            }

            // Find contours from the segmented output
            cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Draw contours on a copy of the original frame
            boundingFrame = frame.clone();
            for (size_t i = 0; i < contours.size(); i++) {
                cv::drawContours(boundingFrame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Live - Contours", boundingFrame);
            videoboundingbox.write(boundingFrame);

            // Compute features for contours
            vector<vector<float>> features = computeAndDisplayFeatures(boundingFrame, contours);

            // Classify features and visualize
            for (const auto& feature : features) {
                string classified_class = classifyKFeatureVector(feature, known_features, class_names, std_devs,k);
                // Visualize the classified class on the video stream
                // Assume 'centroid' can be computed from the 'feature'
                Point centroid = Point(static_cast<int>(feature[1]), static_cast<int>(feature[2]));
                putText(frame, classified_class, centroid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            }

            // Show the frame with the classification
            cv::imshow("Live Feed - Classified Objects", frame);

            // Break the loop if the user presses 'q' or ESC
            char key = cv::waitKey(30);
            if (key == 'q' || key == 27) { // ESC key or 'q' key to quit
                break;
            }
        }
        }         
}
            char key = cv::waitKey(30);
            if (key == 'q' || key == 27) { // ESC key or 'q' key to quit
                break;
}
    }
videoBlurredlv.release();
videoDarkenedlv.release();
videoThresholdedlv.release();
videoCleanedlv.release();
videoGrabcutlv.release();
videoSegmentedlv.release();
videoSegmented33.release();
videoboundingbox.release();
cap.release();
cv::destroyAllWindows();
}