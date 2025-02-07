#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace cv;
using namespace std;
using namespace dnn;

class ObjectDetectorTracker {
private:
    dnn::Net net;
    vector<string> classes;
    vector<Ptr<Tracker>> trackers;
    vector<Rect> tracking_boxes;  // Changed from Rect2d to Rect
    vector<string> tracking_labels;
    int skip_frames = 0;
    const int max_skip_frames = 30;

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    int inpWidth = 416;
    int inpHeight = 416;

    vector<String> getOutputsNames(const Net& net) {
        static vector<String> names;
        if (names.empty()) {
            vector<int> outLayers = net.getUnconnectedOutLayers();
            vector<String> layersNames = net.getLayerNames();
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }

    void drawPred(Mat& frame, const string& label, float conf, int left, int top, int right, int bottom) {
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);

        string labelWithConf = label + ": " + to_string(conf).substr(0, 4);
        int baseLine;
        Size labelSize = getTextSize(labelWithConf, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        putText(frame, labelWithConf, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

public:
    ObjectDetectorTracker(const string& modelConfig, const string& modelWeights, const string& classesFile) {
        net = readNetFromDarknet(modelConfig, modelWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line)) classes.push_back(line);
    }

    Mat detect(Mat& frame) {
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        net.setInput(blob);
        
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        trackers.clear();
        tracking_boxes.clear();
        tracking_labels.clear();

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            
            Ptr<Tracker> tracker = TrackerCSRT::create();
            tracker->init(frame, box);  // Using Rect instead of Rect2d
            trackers.push_back(tracker);
            tracking_boxes.push_back(box);
            tracking_labels.push_back(classes[classIds[idx]]);

            drawPred(frame, classes[classIds[idx]], confidences[idx],
                    box.x, box.y, box.x + box.width, box.y + box.height);
        }

        return frame;
    }

    Mat track(Mat& frame) {
        if (skip_frames >= max_skip_frames) {
            frame = detect(frame);
            skip_frames = 0;
        } else {
            for (size_t i = 0; i < trackers.size(); ++i) {
                Rect bbox;  // Changed from Rect2d to Rect
                if (trackers[i]->update(frame, bbox)) {
                    tracking_boxes[i] = bbox;
                    rectangle(frame, bbox, Scalar(255, 0, 0), 2);
                    putText(frame, tracking_labels[i], Point(bbox.x, bbox.y - 5),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
                }
            }
            skip_frames++;
        }
        return frame;
    }
};

int main() {
    try {
        ObjectDetectorTracker detector(
            "yolov4.cfg",
            "yolov4.weights",
            "coco.names"
        );

        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cout << "Error opening camera" << endl;
            return -1;
        }

        Mat frame;
        namedWindow("Object Detection and Tracking", WINDOW_AUTOSIZE);

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            frame = detector.track(frame);

            imshow("Object Detection and Tracking", frame);

            if (waitKey(1) == 'q') break;
        }

        cap.release();
        destroyAllWindows();

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
