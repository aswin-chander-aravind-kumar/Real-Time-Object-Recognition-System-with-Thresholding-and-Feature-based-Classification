/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Header file for implementation of a K-means algorithm
*/


#include <opencv2/opencv.hpp>
#include <string>

#ifndef KMEANS_H
#define KMEANS_H

#define SSD(a, b) ( ((int)a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) )

void kmeans(const std::vector<cv::Vec3b>& data, std::vector<cv::Vec3b>& means, std::vector<int>& labels, int K, int maxIterations, double epsilon);


#endif
