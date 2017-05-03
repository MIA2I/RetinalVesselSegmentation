import sys
sys.path.append('/home/john/caffe-master/python/')
sys.path.append('/usr/lib/python2.7/dist-packages/')
sys.path.append('/usr/lib/python2.7/site-packages/')
sys.path.append('/home/john/opencv-3.0.0/build/lib')

import caffe

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

imageID = 14
iterations = ['50000']

for iteration_index in range(1):

    iteration = iterations[iteration_index]

    model = 'deploy.prototxt'
    weights = './AllModels/' + str(imageID) + '/train_iter_' + iteration + '.caffemodel'

    net = caffe.Net(model, weights, caffe.TEST)

    num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
        , '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

    for index in range(20):

        if index is not imageID-1:
            continue

        print iteration, index

        im = Image.open('../Raw/image/' + num[index] + '.png')

        mask = Image.open('../Raw/mask/' + num[index] + '.png')
        mask = np.array(mask, dtype=np.uint8)
        mask[np.where(mask > 0)] = 1

        img = np.array(im, dtype=np.float32)

        image = np.zeros([640, 768])
        image[:,:] = img[:, :]

        segmentation = np.zeros([640, 768])

        for i in range(4):

            for j in range(5):
                in_ = np.zeros([256, 256])

                in_[:, :] = image[i * 128:256 + i * 128, j * 128:256 + j * 128]


                in_ -= np.array((70.9130), dtype=np.float32)
                in_ = in_[np.newaxis, ...]

                net.blobs["data1"].data[...] = in_

                net.forward()

                data = net.blobs["Label1"].data

                label = data[0, 1, ::-1]
                #np.add(label, data[0, 0, ::-1], label)

                label = np.flipud(label)

                MASK = np.ones([256, 256])
                MASK[:, 0:20] = 0
                MASK[:, 236:256] = 0

                if (i == 0):
                    MASK[:, 2:50] = 1

                if (i == 3):
                    MASK[:, 236:256] = 1

                MASK[0:20, :] = 0
                MASK[236:256, :] = 0

                if (j == 0):
                    MASK[0:20, :] = 1

                if (j == 3):
                    MASK[236:256, :] = 1

                np.multiply(MASK, label, label)

                TEMP = segmentation[i * 128: 256 + i * 128, j * 128: 256 + j * 128]
                TEMP[np.where(label > 0)] = 0
                segmentation[i * 128: 256 + i * 128, j * 128: 256 + j * 128] = TEMP

                segmentation[i * 128: 256 + i * 128, j * 128: 256 + j * 128] += label

        pred = segmentation[:,:]

        np.multiply(mask, pred, pred)


        image = Image.fromarray(np.uint8(pred*255))
        image.save('./AllLabels/' +iteration + '/' + num[index] + '_label.png')

        gt = Image.open('../Raw/Label/' + num[index] + '.png')
        gt.save('./AllLabels/' +iteration + '/' + num[index] + '_manual1.png')

        msk = Image.fromarray(np.uint8(mask * 255))
        msk.save('./AllLabels/' +iteration + '/' + num[index] + '_test_mask.png')

