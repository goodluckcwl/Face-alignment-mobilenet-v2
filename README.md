
# DeepAlignment
Face Alignment by MobileNetv2. Note that MTCNN is used to provided the input bbox. You need to modify the path of images in order to run the demo. 
The structure of mobilenetv2 is similar to that of https://github.com/tensor-yu/cascaded_mobilenet-v2.

## Training
The training data including:
- Training data of 300W dataset
- Training data of Menpo dataset
### Data Augmentation
Data augment is important to the performance of face alignment. I have tried several kinds of data augment method, including:
- Random Flip.
- Random Shift.
- Random Scale.
- Random Rotation. The image is rotated by the degree sampled from -30 to 30.
- Random Noise. Gaussian noise is added to the input images.

## Performance
The performance on 300W is not good enough. May be I need to try more times. If you have any ideas, please contact me or open an issue.

|Method|Input Size|Common|Challenge|Full set|Training Data|
|------|------|------|------|------|------|
|VGG-Shadow(With Dropout)|70 * 60|5.66|10.82|6.67|300W|
|Mobilenet-v2-stage1|64 * 64|6.07|10.60|6.96|300W and Menpo|
|Mobilenet-v2-stage2|64 * 64|5.76|8.93|6.39|300W and Menpo|

## Result on 300W
![](https://github.com/goodluckcwl/DeepAlignment/raw/master/sample.jpg)
The ground truth landmarks is donated by white color while the predicted ones blue.

## Pre-train Models
The pre-train models can be downloaded from [baiduyun](https://pan.baidu.com/s/1wYycQbmz3CxBQw9KgJkxEA).

## Demo
I write a demo to view the alignment results.
To run the domo, please do:
1. Download and compile [caffe](https://github.com/goodluckcwl/caffe-windows-ms). Compile pycaffe.
2. Modified the path in demo.py.
3. Run.

## Dependence
To use my code to reproduce the results, you need to use my [caffe](https://github.com/goodluckcwl/caffe-windows-ms). I have added some useful layers.
- [My PyCaffe](https://github.com/goodluckcwl/caffe-windows-ms)
- Opencv
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)


## Reference:
- [Cascaded-Mobilenet-v2](https://github.com/tensor-yu/cascaded_mobilenet-v2)
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
