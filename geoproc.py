#!coding:utf-8

from osgeo import gdal, ogr, osr
import numpy as np

gdtype_dict = {\
    np.dtype("uint8") : gdal.GDT_Byte
    }

def saveGeoTiff(img,savepath,geoTransform,SpaRef):
    cols = img.shape[1]
    rows = img.shape[0]
    gdtype = gdtype_dict[img.dtype]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(savepath, cols, rows, img.shape[2], gdtype)
    outRaster.SetGeoTransform(geoTransform)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(SpaRef)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    if len(img.shape) == 2:
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(img)
        outband.FlushCache()
    else:
        for i in range(0,img.shape[2]):
            outband = outRaster.GetRasterBand(i+1)
            outband.WriteArray(img[:,:,img.shape[2]-1-i])
            outband.FlushCache()

def ReprojectCoords(coords,src_srs,tgt_srs):
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    if not isinstance(coords[0],list) and not isinstance(coords[0],tuple):
        x, y, z  = transform.TransformPoint(coords[0],coords[1])
        return x, y
    else:
        trans_coords = []
        for x, y in coords:
            x, y, z = transform.TransformPoint(x, y)
            trans_coords.append([x, y])
        return trans_coords

def getCoords(geoTransform,points):
    if (not isinstance(points[0],list)) and (not isinstance(points[0],tuple)):
        x = geoTransform[0]+(float(points[0])*geoTransform[1])+(float(points[1])*geoTransform[2])
        y = geoTransform[3] + (float(points[0]) * geoTransform[4]) + (float(points[1]) * geoTransform[5])
        return x, y
    else:
        coords = []
        for px, py in points:
            x = geoTransform[0] + (float(px) * geoTransform[1]) + (float(py) * geoTransform[2])
            y = geoTransform[3] + (float(px) * geoTransform[4]) + (float(py) * geoTransform[5])
            coords.append([x,y])
        return coords

def savePointshapefile(points,layername,projection,savepath):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(savepath)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    layer = data_source.CreateLayer(layername, srs, ogr.wkbPoint)

    for point in points:
        feature = ogr.Feature(layer.GetLayerDefn())
        wkt = "POINT(%f %f)" % (float(point[0]), float(point[1]))
        point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
        feature.Destroy()

    data_source.Destroy()



if __name__ == "__main__":
    imgpath = "C:/work/vehicle_detection/images/raw/ionfukuoka_Z19.tif"
    gimg = gdal.Open(imgpath)
    SpaRef = gimg.GetProjection()
    geoTransform = gimg.GetGeoTransform()
    coords = getCoords(geoTransform,[[0,0],[500,500]])
    print(coords)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(SpaRef)
    tgt_srs = src_srs.CloneGeogCS()
    latlons = ReprojectCoords(coords,src_srs,tgt_srs)
    print(latlons)
    shppath = "C:/work/vehicle_detection/images/raw/ionfukuoka_Z19.shp"
    savePointshapefile(latlons,"vehicles",tgt_srs.ExportToWkt(),shppath)

