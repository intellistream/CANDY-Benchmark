#!/usr/bin/env python3
import sys
#Assuming you have PyCandyxxx.so at ../../../
sys.path.append('../../../')
import torch
import PyCANDY as candy

idxFlat=candy.createIndex('flatSSDGPU')
cfg={'vecDim':4,'metricType':"IP","SSDBufferSize":4}
idxFlat.setConfig(candy.dictToConfigMap(cfg))
idxFlat.startHPC()
a = torch.rand(4,4)
idxFlat.insertTensor(a)
a2= torch.rand(4,4)
idxFlat.insertTensor(a2)
b=a[0:1]
ru=idxFlat.searchTensor(b,2)
print(ru)
idxFlat.endHPC()
