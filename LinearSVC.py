import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from osgeo import gdal, gdal_array

gdal.UseExceptions()
gdal.AllRegister()

def main():

    planta = np.zeros((1, 60, 6))

    planta[:, :, 0], planta[:, :, 1], planta[:, :, 2], planta[:, :, 3], planta[:, :, 4], planta[:, :, 5] = np.genfromtxt(
        'C:\Projeto\Amostras\Bandas\planta_0.csv'), np.genfromtxt(
            'C:\Projeto\Amostras\Bandas\planta_1.csv'), np.genfromtxt(
                'C:\Projeto\Amostras\Bandas\planta_2.csv'), np.genfromtxt(
                    'C:\Projeto\Amostras\Bandas\planta_3.csv'), np.genfromtxt(
                        'C:\Projeto\Amostras\Bandas\planta_4.csv'), np.genfromtxt(
                            'C:\Projeto\Amostras\Bandas\planta_5.csv')

    solo = np.zeros((1, 60, 6))

    solo[:, :, 0], solo[:, :, 1], solo[:, :, 2], solo[:, :, 3], solo[:, :, 4], solo[:, :, 5] = np.genfromtxt(
        'C:\Projeto\Amostras\Bandas\solo_0.csv'), np.genfromtxt(
            'C:\Projeto\Amostras\Bandas\solo_1.csv'), np.genfromtxt(
                'C:\Projeto\Amostras\Bandas\solo_2.csv'), np.genfromtxt(
                    'C:\Projeto\Amostras\Bandas\solo_3.csv'), np.genfromtxt(
                        'C:\Projeto\Amostras\Bandas\solo_4.csv'), np.genfromtxt(
                            'C:\Projeto\Amostras\Bandas\solo_5.csv')
    
    sombra = np.zeros((1, 60, 6))

    sombra[:, :, 0], sombra[:, :, 1], sombra[:, :, 2], sombra[:, :, 3], sombra[:, :, 4], sombra[:, :, 5] = np.genfromtxt(
        'C:\Projeto\Amostras\Bandas\sombra_0.csv'), np.genfromtxt(
            'C:\Projeto\Amostras\Bandas\sombra_1.csv'), np.genfromtxt(
                'C:\Projeto\Amostras\Bandas\sombra_2.csv'), np.genfromtxt(
                    'C:\Projeto\Amostras\Bandas\sombra_3.csv'), np.genfromtxt(
                        'C:\Projeto\Amostras\Bandas\sombra_4.csv'), np.genfromtxt(
                            'C:\Projeto\Amostras\Bandas\sombra_5.csv')
    
    planta_novo = (planta.shape[0] * planta.shape[1], planta.shape[2])
    treino_planta = planta[:, :, :6].reshape(planta_novo)

    solo_novo = (solo.shape[0] * solo.shape[1], solo.shape[2])
    treino_solo = solo[:, :, :6].reshape(solo_novo)

    sombra_novo = (sombra.shape[0] * sombra.shape[1], sombra.shape[2])
    treino_sombra = sombra[:, :, :6].reshape(sombra_novo)

    treino_x = np.zeros((treino_planta.shape[0] + treino_solo.shape[0] + treino_sombra.shape[0], treino_planta.shape[1]))
    treino_x[:60, :], treino_x[60:120, :], treino_x[120:180, :] = treino_planta, treino_solo, treino_sombra

    treino_y = np.zeros((treino_planta.shape[0] + treino_solo.shape[0] + treino_sombra.shape[0], ))
    treino_y[0:59], treino_y[60:119], treino_y[120:179] = [0], [1], [2]

    modelo = LinearSVC(max_iter=10000000)

    modelo.fit(treino_x, treino_y)

    dataset = gdal.Open('C:\Projeto\Resultados\RASTER_MESCLADO_1.tif', gdal.GA_ReadOnly)
    img = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(dataset.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = dataset.GetRasterBand(b + 1).ReadAsArray()
    
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :6].reshape(new_shape)

    classificacao = modelo.predict(X)
    classificacao = classificacao.reshape(img[:, :, 0].shape)

    ds = gdal.Open('C:\Projeto\Resultados\RASTER_MESCLADO_1.tif')
    band = ds.GetRasterBand(6)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    outDataRaster = driver.Create('C:\Projeto\Resultados\CLASSIFICACAO_LINEAR.tif', rows, cols, 1, gdal.GDT_Byte)
    outDataRaster.SetGeoTransform(ds.GetGeoTransform())
    outDataRaster.SetProjection(ds.GetProjection())
    outDataRaster.GetRasterBand(1).WriteArray(classificacao)
    outDataRaster.FlushCache()
    del outDataRaster


if __name__ == '__main__':
    main()