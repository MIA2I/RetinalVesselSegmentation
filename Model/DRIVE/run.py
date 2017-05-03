import sys
sys.path.append('/home/john/caffe-master/python/')
sys.path.append('/usr/lib/python2.7/dist-packages/')
sys.path.append('/usr/lib/python2.7/site-packages/')
sys.path.append('/home/john/opencv-3.0.0/build/lib')

import caffe

import numpy as np
import os

# init
caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')
solver.solve()

#train using command line
#command = '/home/john/caffe-master/build/tools/caffe train -solver /home/john/MIPData/Segmentation/DRIVE/Model/TrainedModel/backup/solver.prototxt -gpu all'
#os.system(command)

# resume training
#command = '/home/john/caffe-master/build/tools/caffe train -solver /home/john/MIPData/Segmentation/DRIVE/Model/TrainedModel/backup5/solver.prototxt -snapshot /home/john/MIPData/Segmentation/DRIVE/Model/TrainedModel/backup5/train_iter_6000.solverstate'
#os.system(command)

# lamda is set to 0.1 since 1000 iterations
