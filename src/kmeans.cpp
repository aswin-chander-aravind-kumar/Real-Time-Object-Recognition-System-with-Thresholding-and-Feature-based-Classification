#include "kmeans.h"
#include <limits>
#include <random>
#include <cmath>

// Helper function to calculate the squared distance between two colors
inline double distance(const cv::Vec3b& a, const cv::Vec3b& b) {
    return std::sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) + std::pow(a[2] - b[2], 2));
}

// Performs the K-means clustering on color pixels
void kmeans(const std::vector<cv::Vec3b>& data, std::vector<cv::Vec3b>& means, std::vector<int>& labels, int K, int maxIterations, double epsilon) {
    int dataSize = data.size();
    labels.resize(dataSize);
    means.resize(K);

    // Randomly initialize cluster centers (means)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dataSize - 1);

    for (int i = 0; i < K; ++i) {
        means[i] = data[dis(gen)];
    }

    bool changed = true;
    int iterations = 0;

    while (changed && iterations < maxIterations) {
        changed = false;
        // E-step: assign points to the nearest cluster
        for (int i = 0; i < dataSize; ++i) {
            double minDist = std::numeric_limits<double>::max();
            int bestCluster = 0;
            for (int k = 0; k < K; ++k) {
                double dist = distance(data[i], means[k]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = k;
                }
            }
            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
        }

        // M-step: update means
        std::vector<cv::Vec3d> newMeans(K, cv::Vec3d(0, 0, 0));
        std::vector<int> counts(K, 0);
        for (int i = 0; i < dataSize; ++i) {
            newMeans[labels[i]] += cv::Vec3d(data[i]);
            counts[labels[i]]++;
        }

        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                means[k] = cv::Vec3b(newMeans[k] / counts[k]);
            }
        }

        ++iterations;
    }
}
