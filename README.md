# Face-alignment-mobilenet-v2
Face Alignment by [MobileNetv2](https://arxiv.org/abs/1801.04381). Note that MTCNN is used to provided the input boundingbox. You need to modify the path of images in order to run the demo. 

## Network Structure
The most important part of the mobilenet-v2 network is the design of bottleneck. In our experiments, we crop the face image by the boundingbox and resize it to <a href="https://www.codecogs.com/eqnedit.php?latex=64&space;\times&space;64" target="_blank"><img src="https://latex.codecogs.com/gif.latex?64&space;\times&space;64" title="64 \times 64" /></a>, which is the input size of the network. Based on this, we can design the structure of our customized mobilenet-v2 for facial landmark lacalization. Note that the receptive field is a key factor to the design of the network.

|Input|Operator|t|channels|n|stride|
|------|------|------|------|------|------|
|<a href="https://www.codecogs.com/eqnedit.php?latex=64^2&space;\times&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?64^2&space;\times&space;3" title="64^2 \times 3" /></a>|conv2d|-|16|1|2|
|<a href="https://www.codecogs.com/eqnedit.php?latex=32^2&space;\times&space;16" target="_blank"><img src="https://latex.codecogs.com/gif.latex?32^2&space;\times&space;16" title="32^2 \times 16" /></a>|bottleneck|6|24|1|2|
|<a href="https://www.codecogs.com/eqnedit.php?latex=16^2&space;\times&space;24" target="_blank"><img src="https://latex.codecogs.com/gif.latex?16^2&space;\times&space;24" title="16^2 \times 24" /></a>|conv2d|6|24|1|1|
|<a href="https://www.codecogs.com/eqnedit.php?latex=16^2&space;\times&space;24" target="_blank"><img src="https://latex.codecogs.com/gif.latex?16^2&space;\times&space;24" title="16^2 \times 24" /></a>|conv2d|6|32|1|2|
|<a href="https://www.codecogs.com/eqnedit.php?latex=8^2&space;\times&space;32" target="_blank"><img src="https://latex.codecogs.com/gif.latex?8^2&space;\times&space;32" title="8^2 \times 32" /></a>|conv2d|6|32|1|1|
|<a href="https://www.codecogs.com/eqnedit.php?latex=8^2&space;\times&space;32" target="_blank"><img src="https://latex.codecogs.com/gif.latex?8^2&space;\times&space;32" title="8^2 \times 32" /></a>|conv2d|6|64|1|2|
|<a href="https://www.codecogs.com/eqnedit.php?latex=4^2&space;\times&space;64" target="_blank"><img src="https://latex.codecogs.com/gif.latex?4^2&space;\times&space;64" title="4^2 \times 64" /></a>|conv2d|6|64|1|1|
|<a href="https://www.codecogs.com/eqnedit.php?latex=4^2&space;\times&space;64" target="_blank"><img src="https://latex.codecogs.com/gif.latex?4^2&space;\times&space;64" title="4^2 \times 64" /></a>|inner product|-|200|1|-|
|200|inner product|-|200|1|-|
|200|inner product|-|50|1|-|
|50|inner product|-|136|1|-|

Note that this structure mainly has two features:
 - Use LeakyReLU rather than ReLU.
 - Use bottleneck embedding, which is 50 in our experiments.

## Training
The training data including:
- Training data of 300W dataset
- Training data of Menpo dataset
### Data Augmentation
Data augmentation is important to the performance of face alignment. I have tried several kinds of data augmentation method, including:
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

## Dataset

|Dataset|Number of images for training|
|------|-----|
|300-W|3148|
|Menpo|12006|

## Result on 300W
![](https://github.com/goodluckcwl/DeepAlignment/raw/master/sample.jpg)
The ground truth landmarks is donated by white color while the predicted ones blue.

## Pre-train Models
The pre-train models can be downloaded from [baiduyun](https://pan.baidu.com/s/1wYycQbmz3CxBQw9KgJkxEA) or [GoogleDisk](https://drive.google.com/open?id=1Nw9s4sZ5tyS6MDMm-DUIdIrSEgw-kH0B).

## Demo
I write a demo to view the alignment results. Besides, the yaw, row and pitch parameters are estimated by the predicted landmarks.
To run the domo, please do:
1. Download and compile [caffe](https://github.com/goodluckcwl/custom-caffe). Compile pycaffe.
2. Use MTCNN to detect face of the images and save the boundingbox of faces.
3. Modified the path in demo.py.
4. Run.
![](https://github.com/goodluckcwl/DeepAlignment/raw/master/demo.png)

## Dependence
To use my code to reproduce the results, you need to use my [caffe](https://github.com/goodluckcwl/custom-caffe). I have added some useful layers.
- [My PyCaffe](https://github.com/goodluckcwl/custom-caffe)
- Opencv
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)


## Reference:
- [Cascaded-Mobilenet-v2](https://github.com/tensor-yu/cascaded_mobilenet-v2)
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- [Mobilenet-v2](https://arxiv.org/abs/1801.04381)
