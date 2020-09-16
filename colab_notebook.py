# -*- coding: utf-8 -*-
"""TP2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VnOlX3xcjpsYnWXY3iZM2_WYEn6usm9R
"""

from google.colab import drive

drive.mount('/content/drive')

!pip install rasterio

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

import rasterio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
# %matplotlib inline

raster = rasterio.open('/content/drive/My Drive/Colab Notebooks/TP2/raster.tif')

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols = 1, nrows = 6, figsize = (70, 70), sharey = True)

show((raster, 1), cmap = 'Greys', ax = ax1)
show((raster, 2), cmap = 'Greys', ax = ax2)
show((raster, 3), cmap = 'Greys', ax = ax3)
show((raster, 4), cmap = 'Greys', ax = ax4)
show((raster, 5), cmap = 'Greys', ax = ax5)
show((raster, 6), cmap = 'Greys', ax = ax6)

ax1.set_title('RED')
ax2.set_title('GREEN')
ax3.set_title('BLUE')
ax4.set_title('WHITE')
ax5.set_title('GLI')
ax6.set_title('MDS')

from osgeo import gdal, gdal_array

gdal.UseExceptions()
gdal.AllRegister()

def exportar(dataset, classificacao):
  
  band = dataset.GetRasterBand(6)
  arr = band.ReadAsArray()
  [cols, rows] = arr.shape

  format = "GTiff"
  driver = gdal.GetDriverByName(format)
  outDataRaster = driver.Create('/content/drive/My Drive/Colab Notebooks/TP2/classificacao.tif', rows, cols, 1, gdal.GDT_Byte)

  outDataRaster.SetGeoTransform(dataset.GetGeoTransform())
  outDataRaster.SetProjection(dataset.GetProjection())

  outDataRaster.GetRasterBand(1).WriteArray(classificacao)
  outDataRaster.FlushCache()
  del outDataRaster


def nao_supervisionado(dataset):

  img = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(dataset.GetRasterBand(1).DataType))

  for b in range(img.shape[2]):
    img[:, :, b] = dataset.GetRasterBand(b + 1).ReadAsArray()
      
  new_shape = (img.shape[0] * img.shape[1], img.shape[2])
  X = img[:, :, :6].reshape(new_shape)

  k_means = cluster.KMeans(n_clusters=4)
  k_means.fit(X)

  X_cluster = k_means.labels_
  classificacao = X_cluster.reshape(img[:, :, 0].shape)
  
  exportar(dataset, classificacao)


def supervisionado(dataset):

  plan = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TP2/planta.csv', sep=';')

  plan.columns = [str(i) for i in range(0, len(plan.columns))]

  planta = np.zeros((1, len(plan.index), len(plan.columns)))

  for i in range(0, len(plan.columns)):
    planta[:, :, i] = plan[str(i)]

  sol = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TP2/solo.csv', sep=';')

  sol.columns = [str(i) for i in range(0, len(sol.columns))]

  solo = np.zeros((1, len(sol.index), len(sol.columns)))

  for i in range(0, len(sol.columns)):
    solo[:, :, i] = sol[str(i)]

  som = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TP2/sombra.csv', sep=';')

  som.columns = [str(i) for i in range(0, len(som.columns))]

  sombra = np.zeros((1, len(som.index), len(som.columns)))

  for i in range(0, len(som.columns)):
    sombra[:, :, i] = som[str(i)]
      
  planta_novo = (planta.shape[0] * planta.shape[1], planta.shape[2])
  treino_planta = planta[:, :, :len(plan.columns)].reshape(planta_novo)

  solo_novo = (solo.shape[0] * solo.shape[1], solo.shape[2])
  treino_solo = solo[:, :, :len(sol.columns)].reshape(solo_novo)

  sombra_novo = (sombra.shape[0] * sombra.shape[1], sombra.shape[2])
  treino_sombra = sombra[:, :, :len(som.columns)].reshape(sombra_novo)

  treino_x = np.zeros((treino_planta.shape[0] + treino_solo.shape[0] + treino_sombra.shape[0], treino_planta.shape[1]))
  treino_x[:len(plan.index), :], treino_x[len(plan.index):len(plan.index) * 2, :], treino_x[len(plan.index) * 2:len(plan.index) * 3, :] = treino_planta, treino_solo, treino_sombra

  treino_y = np.zeros((treino_planta.shape[0] + treino_solo.shape[0] + treino_sombra.shape[0], ))
  treino_y[0:len(plan.index) - 1], treino_y[len(plan.index):len(plan.index) * 2 - 1], treino_y[len(plan.index) * 2:len(plan.index) * 3 - 1] = [0], [1], [2]

  modelo = LinearSVC(max_iter = 10000000)

  modelo.fit(treino_x, treino_y)

  img = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(dataset.GetRasterBand(1).DataType))

  for b in range(img.shape[2]):
    img[:, :, b] = dataset.GetRasterBand(b + 1).ReadAsArray()
      
  new_shape = (img.shape[0] * img.shape[1], img.shape[2])
  X = img[:, :, :img.shape[2]].reshape(new_shape)

  classificacao = modelo.predict(X)
  classificacao = classificacao.reshape(img[:, :, 0].shape)

  exportar(dataset, classificacao)

# /content/drive/My Drive/Colab Notebooks/TP2/raster.tif
dataset = gdal.Open(input('\nCaminho para a imagem:\n\n'), gdal.GA_ReadOnly)

metodo = input('\nMétodo de classificação:\n\n(0) K-means Clustering\n\n(1) Linear Support Vector Classification\n\n')

if metodo == '0':
  nao_supervisionado(dataset)
elif metodo == '1':
  supervisionado(dataset)

classificacao = rasterio.open('/content/drive/My Drive/Colab Notebooks/TP2/classificacao.tif')

fig, ax1 = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 10), sharey = True)

show((classificacao, 1), cmap = 'Greys', ax = ax1)

ax1.set_title('CLASSIFICACAO')

fig, ax1 = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 10), sharey = True)

show_hist((classificacao, 1), bins = 50, histtype = 'stepfilled',
          lw = 0.0, stacked = False, alpha = 0.3, ax = ax1)

ax1.set_title('HISTOGRAMA')

plt.show()