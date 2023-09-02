# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:40:18 2023

@author: ryujih
"""

import numpy as np
import struct

fname='Lon_2km.bin'
with open(fname,'r') as f:
    a=np.fromfile(f,dtype=np.float32)