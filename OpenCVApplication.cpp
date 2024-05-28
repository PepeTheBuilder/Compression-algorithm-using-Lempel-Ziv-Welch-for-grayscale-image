#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <fstream>
#include <bitset>
#include <cmath>

using namespace std;
using namespace cv;

// gri1.bmp <-> gri4.bmp or 1<->13.bmp or Onegray.bmp
constexpr auto PATH_IMG = "Images/gri/gri3.bmp";
constexpr auto PATH_FILE = "compressed_data.bin";
constexpr auto PATH_FILE2 = "compressed_data2.bin";

vector<int> compressLZW(const Mat_<uchar>& image) {
    unordered_map<string, int> dictionary;
    vector<int> compressedData;
    for (int i = 0; i < 256; i++) {
        dictionary[string(1, (uchar)i)] = i;
    }

    string currentString;
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            uchar pixel = image(x, y);
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

vector<uint8_t> compressBinary(const vector<int>& compressedData) {
    int maxValue = *max_element(compressedData.begin(), compressedData.end());
    int bitsRequired = ceil(log2(maxValue + 1));

    vector<uint8_t> binaryData;
    int bitBuffer = 0;
    int bitCount = 0;

    for (int value : compressedData) {
        bitBuffer = (bitBuffer << bitsRequired) | value;
        bitCount += bitsRequired;

        while (bitCount >= 8) {
            bitCount -= 8;
            binaryData.push_back((bitBuffer >> bitCount) & 0xFF);
        }
    }

    if (bitCount > 0) {
        binaryData.push_back((bitBuffer << (8 - bitCount)) & 0xFF);
    }

    return binaryData;
}

void writeBinaryFile(const string& filename, const vector<uint8_t>& binaryData, int bitsRequired) {
    ofstream outFile(filename, ios::binary);

    outFile.write(reinterpret_cast<const char*>(&bitsRequired), sizeof(bitsRequired));

    outFile.write(reinterpret_cast<const char*>(binaryData.data()), binaryData.size());

    outFile.close();
}

vector<int> readBinaryFile(const string& filename) {
    ifstream inFile(filename, ios::binary);

    int bitsRequired;
    inFile.read(reinterpret_cast<char*>(&bitsRequired), sizeof(bitsRequired));

    vector<uint8_t> binaryData((istreambuf_iterator<char>(inFile)), istreambuf_iterator<char>());

    inFile.close();

    vector<int> decompressedData;
    int bitBuffer = 0;
    int bitCount = 0;

    for (uint8_t byte : binaryData) {
        bitBuffer = (bitBuffer << 8) | byte;
        bitCount += 8;

        while (bitCount >= bitsRequired) {
            bitCount -= bitsRequired;
            int value = (bitBuffer >> bitCount) & ((1 << bitsRequired) - 1);
            decompressedData.push_back(value);
        }
    }

    return decompressedData;
}

Mat_<uchar> decompressLZW(const vector<int>& compressedData, int width, int height) {
    unordered_map<int, string> dictionary;
    vector<uchar> decompressedData;

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
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            if (it == decompressedData.end()) {
                throw runtime_error("Decompressed data size mismatch");
            }
            decompressedImage(x, y) = *it++;
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

long getFileSize(const string& filename) {
    ifstream inFile(filename, ios::binary | ios::ate);
    return inFile.tellg();
}

double compressRatio(const string& imagePath, const string& binaryPath) {
    long originalSize = getFileSize(imagePath);

    long compressedSize = getFileSize(binaryPath);

    if (compressedSize == 0) {
        cerr << "compressed file size is zero, cannot calculate compression ratio." << endl;
        return 0;
    }

    double ratio = originalSize / double(compressedSize);
    return ratio;
}

int countDifferentPixels(const Mat& image1, const Mat& image2) {
    if (image1.rows != image2.rows || image1.cols != image2.cols) {
        return -1;
    }

    int differentPixels = 0;
    for (int x = 0; x < image1.rows; x++) {
        for (int y = 0; y < image1.cols; y++) {
            if (image1.at<uchar>(x, y) != image2.at<uchar>(x, y)) {
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

    vector<int> compressedData = compressLZW(image);

    vector<uint8_t> binaryData = compressBinary(compressedData);

    int maxValue = *max_element(compressedData.begin(), compressedData.end());
    int bitsRequired = ceil(log2(maxValue ));

    writeBinaryFile(PATH_FILE, binaryData, bitsRequired);

    //the old way of writing the data (the hole integer)
    saveCompressedData(compressedData, PATH_FILE2);

    vector<int> loadedData = readBinaryFile(PATH_FILE);
    Mat decompressedImage = decompressLZW(loadedData, image.cols, image.rows);

    cout << "Compression ratio: " << compressRatio(PATH_IMG, PATH_FILE) << '\n'; //
    cout << "Different pixels = " << countDifferentPixels(image, decompressedImage) << '\n';

    imshow("original", image);
    imshow("decompressed", decompressedImage);
    waitKey();

    return 0;
}
