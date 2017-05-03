import sys
sys.path.append('/home/john/caffe-master/python/')
sys.path.append('/usr/lib/python2.7/dist-packages/')
sys.path.append('/usr/lib/python2.7/site-packages/')
sys.path.append('/home/john/opencv-3.0.0/build/lib')

import caffe

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

model = 'deploy.prototxt'
weights = 'train_iter_50000.caffemodel'

net = caffe.Net(model, weights, caffe.TEST)

num = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'
    , '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

for index in range(20):

    #if index is not 3:
    #    continue

    print index

    im = Image.open('/home/john/MIPData/Segmentation/DRIVE/Raw/test/image/' + num[index] + '_test.tif')

    mask = Image.open('/home/john/MIPData/Segmentation/DRIVE/Raw/test/mask/' + num[index] + '_test_mask.gif')
    mask = np.array(mask, dtype=np.uint8)
    mask[np.where(mask > 0)] = 1

    img = np.array(im, dtype=np.float32)

    image = np.zeros([640, 640])
    image[28:28 + 584, 37:37 + 565] = img[:, :]

    segmentation = np.zeros([640, 640])

    for i in range(4):

        for j in range(4):
            in_ = np.zeros([256, 256])

            in_[:, :] = image[i * 128:256 + i * 128, j * 128:256 + j * 128]


            in_ -= np.array((66.9831), dtype=np.float32)
            in_ = in_[np.newaxis, ...]

            net.blobs["data1"].data[...] = in_

            net.forward()

            data = net.blobs["Label1"].data

            label = data[0, 1, ::-1]
            #np.add(label, data[0, 0, ::-1], label)

            label = np.flipud(label)

            MASK = np.ones([256, 256])
            MASK[:, 0:50] = 0
            MASK[:, 206:256] = 0

            if (i == 0):
                MASK[:, 0:50] = 1

            if (i == 3):
                MASK[:, 206:256] = 1

            MASK[0:50, :] = 0
            MASK[206:256, :] = 0

            if (j == 0):
                MASK[0:50, :] = 1

            if (j == 3):
                MASK[206:256, :] = 1

            np.multiply(MASK, label, label)

            TEMP = segmentation[i * 128: 256 + i * 128, j * 128: 256 + j * 128]
            TEMP[np.where(label > 0)] = 0
            segmentation[i * 128: 256 + i * 128, j * 128: 256 + j * 128] = TEMP

            segmentation[i * 128: 256 + i * 128, j * 128: 256 + j * 128] += label

    pred = segmentation[28:28 + 584, 37:37 + 565]

    np.multiply(mask, pred, pred)


    image = Image.fromarray(np.uint8(pred*255))
    image.save('./labels/50000/' + num[index] + '_label.png')

    gt = Image.open('/home/john/MIPData/Segmentation/DRIVE/Raw/test/1st_manual/' + num[index] + '_manual1.gif')
    gt.save('./labels/50000/' + num[index] + '_manual1.gif')

    msk = Image.fromarray(np.uint8(mask * 255))
    msk.save('./labels/50000/' + num[index] + '_test_mask.gif')

