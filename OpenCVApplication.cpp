#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <fstream>

using namespace std;
using namespace cv;

// gri1.jpg <-> gri4.jpg or 1<->13.bmp or Onegray.png
constexpr auto PATH_IMG = "Images/gri/Onegray.png";
constexpr auto PATH_FILE = "compressed_data.bin";

vector<int> compressLZW(const Mat_<uchar>& image) {
    unordered_map<string, int> dictionary;
    vector<int> compressedData;

    for (int i = 0; i < 256; ++i) {
        dictionary[string(1, (uchar)i)] = i;
    }

    string currentString;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            uchar pixel = image(y, x);
            string nextString = currentString + (char)pixel;
            if (dictionary.find(nextString) != dictionary.end()) {
                currentString = nextString;
            }
            else {
                compressedData.push_back(dictionary[currentString]);
                dictionary[nextString] = dictionary.size();
                currentString = string(1, (char)pixel);
            }
        }
    }
    if (!currentString.empty()) {
        compressedData.push_back(dictionary[currentString]);
    }

    return compressedData;
}

Mat_<uchar> decompressLZW(const vector<int>& compressedData, int width, int height) {
    unordered_map<int, string> dictionary;
    vector<uchar> decompressedData;

    // Initialize dictionary with grayscale pixel values
    for (int i = 0; i < 256; ++i) {
        dictionary[i] = string(1, (uchar)i);
    }

    string currentString = dictionary[compressedData[0]];
    decompressedData.insert(decompressedData.end(), currentString.begin(), currentString.end());
    string nextString;
    for (size_t i = 1; i < compressedData.size(); ++i) {
        int code = compressedData[i];
        if (dictionary.find(code) != dictionary.end()) {
            nextString = dictionary[code];
        }
        else if (code == dictionary.size()) {
            nextString = currentString + currentString[0];
        }
        else {
            throw runtime_error("Invalid compressed data");
        }
        decompressedData.insert(decompressedData.end(), nextString.begin(), nextString.end());
        dictionary[dictionary.size()] = currentString + nextString[0];
        currentString = nextString;
    }

    Mat_<uchar> decompressedImage(height, width);
    auto it = decompressedData.begin();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (it == decompressedData.end()) {
                throw runtime_error("Decompressed data size mismatch");
            }
            decompressedImage(y, x) = *it++;
        }
    }

    return decompressedImage;
}

void saveCompressedData(const vector<int>& compressedData, const string& filename) {
    ofstream file(filename, ios::binary);
    if (file.is_open()) {
        for (int code : compressedData) {
            file.write(reinterpret_cast<const char*>(&code), sizeof(code));
        }
        file.close();
    }
    else {
        cerr << "Unable to open file for writing: " << filename << endl;
    }
}

vector<int> loadCompressedData(const string& filename) {
    vector<int> compressedData;
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int code;
        while (file.read(reinterpret_cast<char*>(&code), sizeof(code))) {
            compressedData.push_back(code);
        }
        file.close();
    }
    else {
        cerr << "Unable to open file for reading: " << filename << endl;
    }
    return compressedData;
}

double calculateCompressionRatio(const Mat& original, const vector<int>& compressedData) {
    int originalSize = original.rows * original.cols;
    int compressedSize = compressedData.size();
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

int countDifferentPixels(const Mat& image1, const Mat& image2) {
    if (image1.rows != image2.rows || image1.cols != image2.cols) {
        return -1;
    }

    int differentPixels = 0;
    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            if (image1.at<uchar>(y, x) != image2.at<uchar>(y, x)) {
                ++differentPixels;
            }
        }
    }

    return differentPixels;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    Mat_ <uchar> image = imread(PATH_IMG, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Unable to load image" << endl;
        return -1;
    }

    // Compress image
    vector<int> compressedData = compressLZW(image);

    // Save compressed data to binary file
    saveCompressedData(compressedData, PATH_FILE);

    // Load compressed data from file
    vector<int> loadedData = loadCompressedData(PATH_FILE);

    // Decompress data
    Mat decompressedImage = decompressLZW(loadedData, image.cols, image.rows);

    // Calculate compression ratio
    double compressionRatio = calculateCompressionRatio(image, compressedData);
    cout << "Compression ratio: " << compressionRatio << '\n';


    image = imread(PATH_IMG, IMREAD_GRAYSCALE);
    imshow("original", image);
    imshow("decompressed", decompressedImage);
    waitKey();

    cout << "Different pixels = " << countDifferentPixels(image, decompressedImage)<< '\n';

    return 0;
}

