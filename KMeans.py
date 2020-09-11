import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from osgeo import gdal, gdal_array

gdal.UseExceptions()
gdal.AllRegister()

def main():
    dataset = gdal.Open('C:\Projeto\Resultados\RASTER_MESCLADO_1.tif', gdal.GA_ReadOnly)
    img = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(dataset.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = dataset.GetRasterBand(b + 1).ReadAsArray()
    
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :6].reshape(new_shape)

    k_means = cluster.KMeans(n_clusters=4)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)

    ds = gdal.Open('C:\Projeto\Resultados\RASTER_MESCLADO_1.tif')
    band = ds.GetRasterBand(6)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    outDataRaster = driver.Create('C:\Projeto\Resultados\KMEANS.tif', rows, cols, 1, gdal.GDT_Byte)
    outDataRaster.SetGeoTransform(ds.GetGeoTransform())
    outDataRaster.SetProjection(ds.GetProjection())
    outDataRaster.GetRasterBand(1).WriteArray(X_cluster)
    outDataRaster.FlushCache()
    del outDataRaster


if __name__ == '__main__':
    main()