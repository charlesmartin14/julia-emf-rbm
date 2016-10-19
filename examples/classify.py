#!/usr/bin/env python
import numpy as np
import h5py

with h5py.File('mnistexample_params.h5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())


    b = np.array(hf.get('mnistexample___bias'))
    vb = np.array(hf.get('mnistexample___vbias'))
    w = np.array(hf.get('mnistexample___weight'))
   
    print b.shape, vb.shape, w.shape

    
