# People Counting System in real-time  
People Counting in Real-Time using live video stream/IP camera in OpenCV.

> this is an modification/improvement to https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/ by adrian rosebrock

<div align="center">
<img src=https://imgur.com/tZYiOkt.gif" width=600>
<p>Live demo from an saved ip camera feed</p>
</div>

- This project is build for the Project Deep Blue Season - 6 National level Hackathon with PS: Crowd Counting Challenge.
- Use case: counting the number of people in the stores/buildings/shopping malls etc., in real-time.
- Sending an alert to the staff if the people are way over the limit.
- Automating features and optimising the real-time stream for better performance (with threading).
- Acts as a measure towards footfall analysis and in a way to tackle COVID-19.

## Table of Content

## Simple Theory
**SSD detector:**
- We are using a SSD (Single Shot Detector) with a MobileNet architecture. In general, it only takes a single shot to detect whatever is in an image. That is, one for generating region proposals, one for detecting the object of each proposal. 
- Compared to other 2 shot detectors like R-CNN, SSD is quite fast.
- MobileNet, as the name implies, is a DNN designed to run on resource constrained devices. For example, mobiles, ip cameras, scanners etc.
- Thus, SSD seasoned with a MobileNet should theoretically result in a faster, more efficient object detector.
---
**Centroid tracker:**
- Centroid tracker is one of the most reliable trackers out there.
- To be straightforward, the centroid tracker computes the centroid of the bounding boxes.
- That is, the bounding boxes are (x, y) co-ordinates of the objects in an image. 
- Once the co-ordinates are obtained by our SSD, the tracker computes the centroid (center) of the box. In other words, the center of an object.
- Then an unique ID is assigned to every particular object deteced, for tracking over the sequence of frames.

## Running/Inference
```
pip install -r requirements.txt
```
```
python run.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4
```
> To run inference on an IP camera:
```
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
url = ''
```
- Then run with the command:
```
python run.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel
```
> Set url = 0 for webcam.

## Modification/Features
The following is an example of the added features. Note: You can easily on/off them in the config. options (mylib/config.py):

<img src="https://imgur.com/Lr8mdUW.png" width=500>
