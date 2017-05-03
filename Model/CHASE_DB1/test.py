import sys
sys.path.append('/home/john/caffe-master/python/')
sys.path.append('/usr/lib/python2.7/dist-packages/')
sys.path.append('/usr/lib/python2.7/site-packages/')
sys.path.append('/home/john/opencv-3.0.0/build/lib')

import caffe

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

Iteration = '50000'
model = 'deploy.prototxt'
weights = 'train_iter_' + Iteration + '.caffemodel'

net = caffe.Net(model, weights, caffe.TEST)

num = ['01L', '01R', '02L', '02R', '03L', '03R', '04L', '04R', '05L', '05R'
    , '06L', '06R', '07L', '07R', '08L', '08R', '09L', '09R', '10L', '10R']

for index in range(20):

    #if index is not 3:
    #    continue

    print index

    im = Image.open('/home/john/MIPData/Segmentation/CHASEDB1/Test/image/Image_' + num[index] + '.jpg')

    mask = Image.open('/home/john/MIPData/Segmentation/CHASEDB1/Test/mask/Image_' + num[index] + '_MASK.png')
    mask = np.array(mask, dtype=np.uint8)
    mask[np.where(mask > 0)] = 1

    img = np.array(im, dtype=np.float32)

    image = np.zeros([1024, 1024])
    image[32:32+960, 12:12 + 999] = img[:, :]

    segmentation = np.zeros([1024, 1024])

    for i in range(7):

        for j in range(7):
            in_ = np.zeros([256, 256])

            in_[:, :] = image[i * 128:256 + i * 128, j * 128:256 + j * 128]


            in_ -= np.array((44.3911), dtype=np.float32)
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

    pred = segmentation[32:32+960, 12:12 + 999]

    #np.multiply(mask, pred, pred)


    image = Image.fromarray(np.uint8(pred*255))
    image.save('./labels/' + Iteration + '/' + num[index] + '_label.png')

    gt = Image.open('/home/john/MIPData/Segmentation/CHASEDB1/Test/1st_manual/Image_' + num[index] + '_1stHO.png')
    gt.save('./labels/' + Iteration + '/' + num[index] + '_manual1.gif')

    #msk = Image.fromarray(np.uint8(mask * 255))
    #msk.save('./labels/30000/' + num[index] + '_test_mask.gif')

