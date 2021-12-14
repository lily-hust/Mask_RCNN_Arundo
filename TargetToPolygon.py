# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# TargetToPolygon.py
# Created on: 2021-11-10
#   
# Description:
#
# Author: Yan Li
# ---------------------------------------------------------------------------

# Import arcpy module
import arcpy
import os
import re
from arcpy.sa import *
arcpy.CheckOutExtension("Spatial")

arcpy.env.addOutputsToMap = False
arcpy.env.overwriteOutput = True

ROOT_DIR = "H:\\SpeciesClassification\\arundo\\Mask_RCNN-Lee"
os.chdir(ROOT_DIR)
subset = "test"
DETE_DIR = os.path.join(os.path.join(ROOT_DIR, "detections"), "exp")
IMG_DIR = os.path.join(DETE_DIR, "images")
SHP_DIR = os.path.join(DETE_DIR, "shapes")
inpath = os.path.join(IMG_DIR, subset)
outdir = os.path.join(SHP_DIR, subset)
outname = subset + "_detectedtarget.shp"
outpath = os.path.join(outdir, outname)

#outpath="H:\\SpeciesClassification\\arundo\\Mask_RCNN-Lee\\detections\\shapes\\test\train_detectedtarget.shp"
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
item = os.listdir(inpath)
p=0
for i in item:
    if i.endswith(".tif"):
        pixel = arcpy.Raster(os.path.join(inpath,i))
        med2 = SetNull(pixel==0, 255)
        
        if  p==0:
            arcpy.RasterToPolygon_conversion(med2, outpath)
        else:
            arcpy.RasterToPolygon_conversion(med2, r'in_memory\tmp')
            arcpy.Append_management(r'in_memory\tmp', outpath)
        p += 1



#inpath="H:\\USGSLiDARTrees\\filtered_data\\"        
#item = os.listdir(inpath)
#for i in item:
#    if i.endswith("tif"):
#        pixel = arcpy.Raster(inpath+i)
#        con = Con(IsNull(pixel),0,1)
#        med = FocalStatistics(con,NbrRectangle(5,5), "MEDIAN")
#        med2 = SetNull(med==0, med)
#        arcpy.RasterToPolygon_conversion(med2, r"in_memory\polygon")
#        arcpy.AddField_management(r"in_memory\polygon", "Area", "DOUBLE")
#        arcpy.CalculateField_management(r"in_memory\polygon", "Area", exp, "PYTHON_9.3")
#        arcpy.Select_analysis(r"in_memory\polygon", outpath+i.replace("_filtered_trees.tif",".shp"), where_clause)
