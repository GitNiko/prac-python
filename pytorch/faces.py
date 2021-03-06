from __future__ import print_function, division
import os
import torch 
import pandas as pd 
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
# warnings.filterwarnings('ignore')
plt.ion()

landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.ix[n, 0]
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
landmarks = landmarks.reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmars):
  plt.imshow(image)
  plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='green')
  plt.pause(100)

plt.figure()
show_landmarks(io.imread(os.path.join('faces/', img_name)), landmarks)

plt.show()