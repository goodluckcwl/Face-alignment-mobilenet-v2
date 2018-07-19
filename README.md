# Face-alignment-mobilenet-v2
Face Alignment by MobileNetv2. Note that MTCNN is used to provided the input bbox. You need to modify the path of images in order to run the demo. 

## Network Structure
The most important part of the mobilenet-v2 network is the design of bottleneck. In our experiments, we crop the face image and resized it to 64*64, which is the input of the network. Based on this, we can design the structure of our customized mobilenet-v2 for facial landmark lacalization.

|Input|Operator|t|channels|n|stride|
|------|------|------|------|------|------|
|$64^2\times 3$|conv2d|-|16|1|2|
|$32^2\times 16$|bottleneck|6|24|1|2|
|$16^2\times 24$|conv2d|6|24|1|1|
|$16^2\times 24$|conv2d|6|32|1|2|
|$8^2\times 32$|conv2d|6|32|1|1|
|$8^2\times 32$|conv2d|6|64|1|2|
|$4^2\times 64$|conv2d|6|64|1|1|
|$4^2\times 64$|inner product|-|200|1|-|
|$200$|inner product|-|200|1|-|
|$200$|inner product|-|50|1|-|
|$50$|inner product|-|136|1|-|

Note that this structure has two features:
 - Use LeakyReLU rather than ReLU.
 - Use bottleneck embedding.

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
1. Download and compile [caffe](https://github.com/goodluckcwl/custom-caffe). Compile pycaffe.
2. Modified the path in demo.py.
3. Run.

![](https://github.com/goodluckcwl/DeepAlignment/raw/master/demo.png)

## Dependence
To use my code to reproduce the results, you need to use my [caffe](https://github.com/goodluckcwl/custom-caffe). I have added some useful layers.
- [My PyCaffe](https://github.com/goodluckcwl/custom-caffe)
- Opencv
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)


## Reference:
- [Cascaded-Mobilenet-v2](https://github.com/tensor-yu/cascaded_mobilenet-v2)
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
