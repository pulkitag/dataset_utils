import numpy as np
import struct
from os import path as osp
from easydict import EasyDict as edict

DATASET_PATH = '/data0/pulkitag/data_sets'

class DataSet(object):
  def get_images(self, setName='train'):
    pass

  def get_labels(self, setName='train'):
    pass

  def get_images_nd_labels(self, setName='train'):
    pass

class MNISTData(DataSet):
  def __init__(self):
    mnistPath  = osp.join(DATASET_PATH, 'mnist')
    self.pths_ = edict()
    self.pths_.train = edict()
    self.pths_.test  = edict()
    self.pths_.train.ims = osp.join(mnistPath, 'train-images-idx3-ubyte')
    self.pths_.train.lb  = osp.join(mnistPath, 'train-labels-idx1-ubyte')
    self.pths_.test.ims = osp.join(mnistPath, 't10k-images-idx3-ubyte')
    self.pths_.test.lb  = osp.join(mnistPath, 't10k-labels-idx1-ubyte')
    
  def get_images(self, setName='train'):
    return self._load_images(self.pths_[setName].ims)

  def get_labels(self, setName='train'):
    return self._load_images(self.pths_[setName].lb)

  def _load_images(self, fileName):
    print (fileName)
    f = open(fileName,'rb')
    magicNum = struct.unpack('>i',f.read(4))
    N = struct.unpack('>i',f.read(4))[0]
    nr = struct.unpack('>i',f.read(4))[0]
    nc = struct.unpack('>i',f.read(4))[0]
    print "Num Images: %d, numRows: %d, numCols: %d" % (N,nr,nc)
    im = np.zeros((N,nr,nc),dtype=np.uint8)
    for i in range(N):
      for r in range(nr):
        for c in range(nc):
          im[i,r,c] = struct.unpack('>B',f.read(1))[0]
    f.close()
    return im

  def _load_labels(self, labelFile):
    f = open(labelFile,'rb')
    magicNum = struct.unpack('>i',f.read(4))
    N = struct.unpack('>i',f.read(4))[0]
    print "Number of labels found: %d" % N
    label = np.zeros((N,1),dtype=np.uint8)
    for i in range(N):
        label[i] = struct.unpack('>B',f.read(1))[0]
    f.close()
    return label

