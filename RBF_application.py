# @authors: Deniz B. Sert, Shawn _____
# with inspiration from the Internet
# @version: February 8, 2021

import numpy as np
import matplotlib.pyplot as plt
from RBF_group3 import RBF_Network
from cv2 import imread
from PIL import Image
import cv2
# from lib6003.image import png_read, show_image


# z = (np.sin(np.sqrt((x-2)**2 + (y-1)**2)) - np.sin(np.sqrt((x+2)**2 + (y+4)**2))) / 2
z = imread('images/small_square.png')
x, y = np.meshgrid(np.linspace(-5, 5, len(z)), np.linspace(-5, 5, len(z)))

print('z shape: ', z.shape)
print('x, y shape: ', x.shape)

print(z.flatten())
print('fitting...')
# fitting RBF-Network with data
features = np.asarray(list(zip(x.flatten(), y.flatten())))
model = RBF_Network(hidden_shape=70, sigma=1)
model.fit(features, z.flatten())
print('fitted.')
print('predicting...')
predictions = model.predict(features)
print('predicted.')



print('plotting...')
# plotting 2D interpolation
figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
figure.suptitle('RBF-Network 2D interpolation', fontsize=20)
axis_right.set_title('fit', fontsize=20)
axis_left.set_title('real', fontsize=20)
axis_left.contourf(x, y, z)
right_image = axis_right.contourf(x, y, predictions.reshape(20, 20))
plt.savefig('2D_test.png')
plt.show()


