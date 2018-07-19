
# coding: utf-8

# This demo is for mobilenet-v2.


import sys
import os
caffe_root = '/home/chenweiliang/caffe-windows-ms'
sys.path.insert(0, caffe_root+'/python')
import caffe
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def preprocess(im, bbox):
    # 添加padding
    pad = cal_padding(bbox, im)
    im_pad = cv2.copyMakeBorder(im, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    bbox = bbox + pad
    # 尺度变换
    bb_w = bbox[2] - bbox[0]
    scale = bb_w * 1.0 / input_width
    h, w, c = im_pad.shape
    # Important
    bbox = bbox / scale
    bbox[0] = round(bbox[0])
    bbox[1] = round(bbox[1])
    bbox[2:] = bbox[0:2] + [input_width - 1, input_height - 1]
    bbox = bbox.astype(np.int32)
    im_pad = cv2.resize(im_pad, (int(w / scale), int(h / scale)))
    cropImg = im_pad[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return cropImg,pad,bbox[0:2]+1,scale

def cal_padding(bbox, im):
    '''
    计算padding的大小
    :param bbox: 
    :param im: 
    :return: 
    '''
    x1,y1,x2,y2 = bbox
    h,w,c = im.shape
    pad = np.max([-x1, -y1, x2 - w, y2 - h, 0]) + 10
    return int(pad)

def pad_bbox(bbox, pad_ratio):
    '''
    添加padding
    :param bbox: 
    :param pad_ratio: 添加
    :return: 
    '''
    # padding
    pad_w = (bbox[2] - bbox[0]) * pad_ratio
    pad_h = (bbox[3] - bbox[1]) * pad_ratio
    bbox = np.array([bbox[0] - pad_w, bbox[1], bbox[2] + pad_w, bbox[3] + 2 * pad_h])
    return np.array(bbox)


def obtain_bbox(bbox, w_h_ratio):
    '''
    生成特定长宽比的bbox
    :param bbox: 
    :param w_h_ratio: 输出的bbox的宽高比
    :return: 
    '''
    bbox = np.array(bbox).astype(np.float32)

    w, h = bbox[2:] - bbox[0:2] + 1
    # 确保高宽比
    if w*1.0/h >= w_h_ratio:
        pad_h = (w/w_h_ratio -h)/2
        pad_w = 0
    elif w/h < w_h_ratio:
        pad_h = 0
        pad_w = (h*w_h_ratio -w)/2
    bbox = bbox[0] - pad_w, bbox[1] - pad_h, bbox[2] + pad_w, bbox[3] + pad_h
    return np.array(bbox)

def rotMatrixToEulerAngle(rotMat):
    theta = cv2.norm(rotMat,cv2.NORM_L2)
    w = np.cos(theta/2);
    x = np.sin(theta/2) * rotMat[0] / theta
    y = np.sin(theta/2) * rotMat[1] / theta
    z = np.sin(theta/2) * rotMat[2] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x * y * z)
    t1 = 1.0 - 2.0 *(x * x + ysqr)
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    elif t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    return roll, yaw, pitch

if __name__ == '__main__':
    caffe.set_mode_cpu()
    # model_def = '/home/chenweiliang/68pts-60-vgg/prototxt/deploy_pool3_conv4_20.prototxt'
    # model_weights = '/home/chenweiliang/68pts-60-vgg/model/drop0.2_iter_59000.caffemodel'
    model_def = '/home/chenweiliang/68pts-64-mobilenetv2/prototxt/deploy.prototxt'
    model_weights = '/home/chenweiliang/68pts-64-mobilenetv2/model/step2_1_iter_600000.caffemodel'
    # model_weights = '/home/chenweiliang/68pts-64-mobilenetv2/model/step4_1_iter_28000.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # h,w,c-> c,h,w

    # 读取图片
    data_dir = 'img'
    imglist = os.listdir(data_dir)
    im_name = 'image_0006.png'
    im_dir = os.path.join(data_dir, im_name)
    im = cv2.imread(im_dir).astype(float)
    # 读取图片对应的人脸框,人脸框由mtcnn检测得到
    with open(im_dir.split('.')[0] + '.txt', 'r') as f:
        line = f.readline()
        tmp = [float(x) for x in line.strip().split(' ')]
        # From matlab indent to C-like indent
        bbox = np.array(tmp[0:4]) - 1
        pts5 = np.array(tmp[4:]) - 1

    # 网络输入图像的大小
    input_width = 64
    input_height = 64

    # 预处理图像
    bbox = np.array(bbox, dtype=np.float32)
    bbox = obtain_bbox(bbox, input_width*1.0/input_height)
    bbox = pad_bbox(bbox, pad_ratio=0.05)

    cropImg, pad, offset, scale = preprocess(im, bbox)
    cv2.imshow('', cropImg.astype(np.uint8))
    # cv2.waitKey(0)

    cropImg = (cropImg - 127.5)/128
    cropImg = transformer.preprocess('data', cropImg)
    net.blobs['data'].data[...] = cropImg
    out = net.forward()['fc4']
    landmarks = out.reshape([2, 68])
    landmarks = np.transpose(landmarks)

    # 反向归一化
    landmarks[:, 0] = landmarks[:, 0] * input_width - 1
    landmarks[:, 1] = landmarks[:, 1] * input_height - 1


    # 使用未重投影的特征点坐标估计相机内参和角度
    focal_length = input_width
    center = [input_width/2, input_height/2]
    camera_matrix = np.zeros([3,3], dtype=np.double)
    camera_matrix[0,:] = [focal_length, 0, center[0]]
    camera_matrix[1,:] = [ 0, focal_length, center[1]]
    camera_matrix[2,:] = [0, 0, 1]
    dist_coeffs = np.zeros([5,1], np.double)
    objectPoints = np.zeros([6,3,1], dtype=np.double)
    objectPoints[0,:,0] = [0,0,0]
    objectPoints[1,:,0] = [0,-330,-65]
    objectPoints[2,:,0] = [-225,170,-135]
    objectPoints[3,:,0] = [225,170,-135]
    objectPoints[4,:,0] = [-150,-150,-125]
    objectPoints[5,:,0] = [150,-150,-125]
    imagePoints = np.zeros([6,2])
    imagePoints = landmarks[[30, 8, 36, 45, 48, 54],:]
    ret, rotVects, transVects = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, dist_coeffs)
    # 转化为欧拉角
    roll, yaw, pitch = rotMatrixToEulerAngle(rotVects)
    print 'roll:%f,yaw:%f,pitch:%f' %(roll,yaw,pitch)

    # 投影到原图, 映射到C-index
    landmarks_ori = (landmarks + offset ) * scale - pad - 1

    # 显示
    im_rgb = np.zeros(im.shape)
    im_rgb[:,:,0] = im[:,:,2]
    im_rgb[:,:,1] = im[:,:,1]
    im_rgb[:,:,2] = im[:,:,0]
    cv2.rectangle(im_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0))
    plt.imshow(im_rgb.astype(np.uint8))
    # Plot
    plt.plot(landmarks_ori[:,0],landmarks_ori[:,1],'r.')
    plt.text(bbox[0],bbox[1],'roll:%f\nyaw:%f\npitch:%f' %(roll,yaw,pitch),fontsize=10,color='w')
    plt.show()




