# -*- coding: utf-8 -*-
"""DeepRare_2019
Authors :
Matei Mancas
Phutphalla Kong
Please check licence when re-using
"""

import numpy as np
from numpy import expand_dims

import cv2

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import keras.backend as K


class DeepRare2019:
  """DeepRare2019 Class.
  """

  def __init__(self):
    """Constructor

    Args:
        face(int): large face detector - disabled : 0, enabled : 1 (default)
        margin (int): adding margins - disabled : 0 (default), enabled : 1 (interesting in images with repeating patterns)
    """

    #self.model = VGG16()

    self._model = (
      VGG16()
    )  # VGG16 is used in this version.
    self._face = (
      1
    )  # by default large faces detector is used
    self._margin = (
      0
    )  # by default additional margins are not added for images (good for classical images)


  def _get_model(self):
    """Read width"""
    return self._model
  def _set_model(self, new_model):
    """Modify witdh"""
    self._model = new_model
  model = property(_get_model, _set_model)

  def _get_margin(self):
    """Read width"""
    return self._margin

  def _set_margin(self, new_margin):
    """Modify witdh"""
    self._margin = new_margin

  margin = property(_get_margin, _set_margin)

  def _get_face(self):
    """Read width"""
    return self._face
  def _set_face(self, new_face):
    """Modify witdh"""
    self._face = new_face
  face = property(_get_face, _set_face)


  def rarity(self, channel):
    """Single-resolution rarity method"""

    channel = np.array(channel)
    a, b = channel.shape

    if a > 50: # manage margins for low-level features
      channel[0:3, :] = 0
      channel[:, a - 3:a] = 0
      channel[:, 0:3] = 0
      channel[b - 3:b, :] = 0

    if a == 28: # manage margins for mid-level features
      channel[0:2, :] = 0
      channel[:, a - 2:a] = 0
      channel[:, 0:2] = 0
      channel[b - 2:b, :] = 0

    if a == 14: # manage margins for high-level features
      channel[0:1, :] = 0
      channel[:, a - 1:a] = 0
      channel[:, 0:1] = 0
      channel[b - 1:b, :] = 0

    channel = cv2.normalize(channel, channel, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

    hist = cv2.calcHist([channel], [0], None, [11], [0, 256])
    hist /= cv2.sumElems(hist)[0]
    hist = -1 * cv2.log(hist + 0.0001) # Histogram rarity

    dst = cv2.calcBackProject([channel], [0], hist, [0, 256], 1) # Back-projection
    dst = cv2.normalize(dst, dst, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    map_max = np.amax(dst)
    map_mean = np.average(dst)
    map_weight = np.square(map_max - map_mean) # Itti-like weight
    dst = dst * map_weight

    return dst


  def apply_rarity(self, layer_output, layer_ind):
    """Apply rarity to full layer"""

    # Begin with fist map from layer_ind
    layer = layer_ind
    feature_maps = layer_output[layer - 1]
    s1, s2, s3, s4 = feature_maps.shape
    map_number = s4

    feature = self.rarity(feature_maps[0, :, :, 0])
    tmp2 = np.power(feature, 2)
    ma = np.max(tmp2)
    me = np.average(tmp2)
    w = (ma - me) * (ma - me)
    tmp2 = w * cv2.normalize(tmp2, tmp2, 0, 255, cv2.NORM_MINMAX)

    # Than add all the other maps of the layer
    for x in range(1, map_number):
      feature = self.rarity(np.copy(feature_maps[0, :, :, x]))
      a, b = feature.shape

      if s2 > 50:
        feature[0:3, :] = 0
        feature[:, a - 3:a] = 0
        feature[:, 0:3] = 0
        feature[b - 3:b, :] = 0
      if s2 == 28:
        feature[0:2, :] = 0
        feature[:, a - 2:a] = 0
        feature[:, 0:2] = 0
        feature[b - 2:b, :] = 0
      if s2 == 14:
        feature[0:2, :] = 0
        feature[:, a - 2:a] = 0
        feature[:, 0:2] = 0
        feature[b - 2:b, :] = 0
      if s2 < 14:
        feature[0:2, :] = 0
        feature[:, a - 2:a] = 0
        feature[:, 0:2] = 0
        feature[b - 2:b, :] = 0

      maps = np.power(feature, 2)
      ma = np.max(maps)
      me = np.average(maps)
      w = (ma - me) * (ma - me)

      tmp2 = (tmp2 + w * cv2.normalize(maps, maps, 0, 255, cv2.NORM_MINMAX)) # maps added based on Itti-like fusion

    return tmp2


  def get_faces(self, layer_output, layer_ind):
    # get layer wich detects large faces
    layer = layer_ind
    feature_maps = layer_output[layer - 1]
    feature = (np.copy(feature_maps[0, :, :, 104]))

    a, b = feature.shape
    feature[0:1, :] = 0
    feature[:, a - 1:a] = 0
    feature[:, 0:1] = 0
    feature[b - 1:b, :] = 0

    return feature


  def fuse_itti(self, mat1, mat2):
    # Itti-like fusion between two maps
    mat1 = cv2.normalize(mat1, mat1, 0, 1, cv2.NORM_MINMAX)
    mat2 = cv2.normalize(mat2, mat2, 0, 1, cv2.NORM_MINMAX)
    m1 = np.amax(mat1)
    a1 = np.average(mat1)
    w1 = np.square(m1 - a1)
    m2 = np.amax(mat2)
    a2 = np.average(mat2)
    w2 = np.square(m2 - a2)
    maps = w1 * mat1 + w2 * mat2
    return maps

  def rarity_network(self, layer_output):
    # compute VGG16 network

    groups = np.zeros((240, 240, 5))

    layer1 = self.apply_rarity(layer_output, 1)
    layer1 = cv2.resize(layer1, (240, 240), interpolation=cv2.INTER_AREA)

    layer2 = self.apply_rarity(layer_output, 2)
    layer2 = cv2.resize(layer2, (240, 240), interpolation=cv2.INTER_AREA)

    layer4 = self.apply_rarity(layer_output, 4)
    layer4 = cv2.resize(layer4, (240, 240), interpolation=cv2.INTER_AREA)

    layer5 = self.apply_rarity(layer_output, 5)
    layer5 = cv2.resize(layer5, (240, 240), interpolation=cv2.INTER_AREA)

    layer7 = self.apply_rarity(layer_output, 7)
    layer7 = cv2.resize(layer7, (240, 240), interpolation=cv2.INTER_AREA)

    layer8 = self.apply_rarity(layer_output, 8)
    layer8 = cv2.resize(layer8, (240, 240), interpolation=cv2.INTER_AREA)

    layer9 = self.apply_rarity(layer_output, 9)
    layer9 = cv2.resize(layer9, (240, 240), interpolation=cv2.INTER_AREA)

    layer11 = self.apply_rarity(layer_output, 11)
    layer11 = cv2.resize(layer11, (240, 240), interpolation=cv2.INTER_AREA)

    layer12 = self.apply_rarity(layer_output, 12)
    layer12 = cv2.resize(layer12, (240, 240), interpolation=cv2.INTER_AREA)

    layer13 = self.apply_rarity(layer_output, 13)
    layer13 = cv2.resize(layer13, (240, 240), interpolation=cv2.INTER_AREA)

    layer15 = self.apply_rarity(layer_output, 15)
    layer15 = cv2.resize(layer15, (240, 240), interpolation=cv2.INTER_AREA)

    layer16 = self.apply_rarity(layer_output, 16)
    layer16 = cv2.resize(layer16, (240, 240), interpolation=cv2.INTER_AREA)

    layer17 = self.apply_rarity(layer_output, 17)
    layer17 = cv2.resize(layer17, (240, 240), interpolation=cv2.INTER_AREA)

    high_level = self.fuse_itti(self.fuse_itti(layer16, layer17), layer15)
    high_level = cv2.normalize(high_level, high_level, 0, 255, cv2.NORM_MINMAX)
    groups[:, :, 4] = high_level

    medium_level2 = self.fuse_itti(self.fuse_itti(layer12, layer13), layer11)
    medium_level2 = cv2.normalize(medium_level2, medium_level2, 0, 255, cv2.NORM_MINMAX)
    groups[:, :, 3] = medium_level2

    medium_level1 = self.fuse_itti(self.fuse_itti(layer8, layer9), layer7)
    medium_level1 = cv2.normalize(medium_level1, medium_level1, 0, 255, cv2.NORM_MINMAX)
    groups[:, :, 2] = medium_level1

    low_level2 = self.fuse_itti(layer4, layer5)
    low_level2 = cv2.normalize(low_level2, low_level2, 0, 255, cv2.NORM_MINMAX)
    groups[:, :, 1] = low_level2

    low_level1 = self.fuse_itti(layer1, layer2)
    low_level1 = cv2.normalize(low_level1, low_level1, 0, 255, cv2.NORM_MINMAX)
    groups[:, :, 0] = low_level1

    SAL = low_level1 + low_level2 + medium_level1 + medium_level2 + high_level

    return SAL, groups

  def compute(self):
    """main method

    Args:
        all.

    Returns:
        m (nparray): image with the final saliency

    """

    if self.img is not None:

      orig_h, orig_w = self.img.shape[:2]
      # RGB image
      self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

      # prepare margins
      if self.margin == 1:
         self.img = cv2.resize(self.img, (168, 168))
         self.img = cv2.copyMakeBorder(self.img, 28, 28, 28, 28, cv2.BORDER_WRAP)

      if self.margin == 0:
         self.img = cv2.resize(self.img, (224, 224))

      # prepare image to put in CNN
      self.img = img_to_array(self.img)  # convert the image to an array
      self.img = expand_dims(self.img, axis=0)  # expand dimensions so that it represents a single 'sample'
      self.img = preprocess_input(self.img)  # prepare the image (e.g. scale pixel values for the vgg)

      # Get all layers without running several times the model
      get_all_layer_outputs = K.function([self.model.layers[0].input], [l.output for l in self.model.layers[1:18]]) # get all layers
      layer_output = get_all_layer_outputs([self.img]) # run all layers on the image

      SAL1, groups1 = self.rarity_network(layer_output)

      # add face if needed
      if self.face == 1:
        face_layer = self.get_faces(layer_output, 15)
        face_resize = cv2.resize(face_layer, (240, 240))

        if self.margin == 1:
          SAL = cv2.resize( SAL1 + face_resize, (224, 224))
          SAL = SAL[30:195, 30:195]

        if self.margin == 0:
          SAL = SAL1 + face_resize

      if self.face == 0:
        if self.margin == 1:
          SAL = cv2.resize( SAL1, (224, 224))
          SAL = SAL[30:195, 30:195]

        if self.margin == 0:
          SAL = SAL1

      SAL = cv2.resize(SAL, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
      SAL = cv2.normalize(SAL, SAL, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

      groups1 = cv2.resize(groups1, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

      face_resize = cv2.resize(face_resize, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

      return SAL, groups1, face_resize #return saliency, groups and face

    else:
      return 0, 0, 0