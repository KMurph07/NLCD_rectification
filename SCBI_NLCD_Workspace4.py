
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:27:04 2018

@author: kmurphy, Bridget Hass, Natalie Robinson
"""

###############################################################################
################# SECTION 1: PULL AND CLEAN TRAINING DATA #####################
###############################################################################

#Import required packages
import h5py
import numpy as np
import pandas as pd
import itertools
import os
import time
import glob
from time import localtime, strftime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from itertools import compress
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.externals import joblib

# InPath where data is pulled
inPath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Extent Reflectance/All/"
outPath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Final/Output/"
workspacePath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Final/Workspace/"
fs_workspacePath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Final/Workspace/Full_Site/"
csvPath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Final/Training/"
broadPath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/"

# Import variable names
# Select number of bands that represent overall spread of wavelengths
# Reduce bands of interest to avoid features with water vapor interference
bands_of_interest = list(range(0,190)) + list(range(212,280)) + list(range(314,416))

# This section is for pulling and cleaning all the bands of the training data 



# Names of .csvs to read in
# Csvs include x and y extents, x and y indicies, and LiDAR canopy height information
DF_csv = "DF_Train.csv" # Deciduous Forest
EF_csv = "EF_Train.csv" # Evergreen Forest
DHI_csv = "DHI_Train.csv" # Developed High Intensity
OW_csv = "OW_Train.csv" # Open Water
PH_csv = "PH_Train.csv" # Pasture Hay
SS_csv = "SS_Train.csv" # Shrub Scrub
DF_CC_csv = "DF_CC_Train.csv" # Deciduous Forest with Cloud Cover
#PH_EX_csv = "PH_EX_Train.csv" # More Pasture Hay to fill out ML algorithm

# Create dataframes showing the location of the training data points
DF_train_loc = pd.read_csv(csvPath+DF_csv)
EF_train_loc = pd.read_csv(csvPath+EF_csv)
DHI_train_loc = pd.read_csv(csvPath+DHI_csv)
OW_train_loc = pd.read_csv(csvPath+OW_csv)
PH_train_loc = pd.read_csv(csvPath+PH_csv)
SS_train_loc = pd.read_csv(csvPath+SS_csv)
DF_CC_train_loc = pd.read_csv(csvPath+DF_CC_csv)
#PH_EX_train_loc = pd.read_csv(csvPath+PH_EX_csv)

# Create column names for the dataframe
bandColNames = []
for i in bands_of_interest:
    bandColNames.append('B' + str(int(i)+1))

# Copy dataframes
DF_training = DF_train_loc.copy()
EF_training = EF_train_loc.copy()
DHI_training = DHI_train_loc.copy()
OW_training = OW_train_loc.copy()
PH_training = PH_train_loc.copy()
SS_training = SS_train_loc.copy()
DF_CC_training = DF_train_loc.copy()
#PH_EX_training = PH_EX_train_loc.copy()

# Add empty band columns
for i in bandColNames:
    DF_training[i]= ""
    EF_training[i]= ""
    DHI_training[i]= ""
    OW_training[i]= ""
    PH_training[i]= ""
    SS_training[i]= ""
    DF_CC_training[i]= ""
    #PH_EX_training[i]= ""

def h5refl2array(refl_filename):
    """h5refl2array reads in a NEON AOP reflectance hdf5 file and returns 
    reflectance array, select metadata, and wavelength dataset.
    --------
    Parameters
        refl_filename -- full or relative path and name of reflectance hdf5 file
    --------
    Returns 
    --------
    reflArray:
        array of reflectance values
    metadata:
        dictionary containing the following metadata:
            EPSG: coordinate reference system code (integer)
            *bad_band_window1: [1340 1445] range of wavelengths to ignore
            *bad_band_window2: [1790 1955] range of wavelengths to ignore 
            ext_dict: dictionary of spatial extent 
            extent: array of spatial extent (xMin, xMax, yMin, yMax)
            mapInfo: string of map information 
            *noDataVal: -9999.0
            projection: string of projection information
            *res: dictionary containing 'pixelWidth' and 'pixelHeight' values (floats)
            *scaleFactor: 10000.0
            shape: tuple of reflectance shape (y, x, # of bands)
        * Asterixed values are the same for all NEON AOP hyperspectral reflectance 
        files processed 2016 & after.
    wavelengths:
        Wavelengths dataset. This is the same for all NEON AOP reflectance hdf5 files.
        wavelengths.value[n-1] gives the center wavelength for band n 
    --------
    This function applies to the NEON hdf5 format implemented in 2016, which 
    applies to data acquired in 2016 & 2017 as of June 2017. Data in earlier 
    NEON hdf5 format is expected to be re-processed after the 2017 flight season. 
    --------
    Example
    --------
    sercRefl, sercRefl_md, wavelengths = h5refl2array('NEON_D02_SERC_DP1_20160807_160559_reflectance.h5') """
    
    #Read in reflectance hdf5 file (include full or relative path if data is located in a different directory)
    hdf5_file = h5py.File(refl_filename,'r')

    #Get the site name
    file_attrs_string = str(list(hdf5_file.items()))
    file_attrs_string_split = file_attrs_string.split("'")
    sitename = file_attrs_string_split[1]
    
    #Extract the reflectance & wavelength datasets
    refl = hdf5_file[sitename]['Reflectance']
    reflArray = refl['Reflectance_Data']
    refl_shape = reflArray.shape
    wavelengths = refl['Metadata']['Spectral_Data']['Wavelength']
    
    #Create dictionary containing relevant metadata information
    metadata = {}
    metadata['shape'] = reflArray.shape
    metadata['mapInfo'] = refl['Metadata']['Coordinate_System']['Map_Info'].value

    #Extract no data value & set no data value to NaN
    metadata['noDataVal'] = float(reflArray.attrs['Data_Ignore_Value'])
    metadata['scaleFactor'] = float(reflArray.attrs['Scale_Factor'])
    
    #Extract bad band windows
    metadata['bad_band_window1'] = (refl.attrs['Band_Window_1_Nanometers'])
    metadata['bad_band_window2'] = (refl.attrs['Band_Window_2_Nanometers'])
    
    #Extract projection information
    metadata['projection'] = refl['Metadata']['Coordinate_System']['Proj4'].value
    metadata['epsg'] = int(refl['Metadata']['Coordinate_System']['EPSG Code'].value)
    
    #Extract map information: spatial extent & resolution (pixel size)
    mapInfo = refl['Metadata']['Coordinate_System']['Map_Info'].value
    mapInfo_string = str(mapInfo); 
    mapInfo_split = mapInfo_string.split(",")
    
    #Extract the resolution & convert to floating decimal number
    metadata['res'] = {}
    metadata['res']['pixelWidth'] = float(mapInfo_split[5])
    metadata['res']['pixelHeight'] = float(mapInfo_split[6])
    
    #Extract the upper left-hand corner coordinates from mapInfo
    xMin = float(mapInfo_split[3]) #convert from string to floating point number
    yMax = float(mapInfo_split[4])
    
    #Calculate the xMax and yMin values from the dimensions
    xMax = xMin + (refl_shape[1]*metadata['res']['pixelWidth']) #xMax = left edge + (# of columns * resolution)",
    yMin = yMax - (refl_shape[0]*metadata['res']['pixelHeight']) #yMin = top edge - (# of rows * resolution)",
    metadata['extent'] = (xMin,xMax,yMin,yMax) #useful format for plotting
    metadata['ext_dict'] = {}
    metadata['ext_dict']['xMin'] = xMin
    metadata['ext_dict']['xMax'] = xMax
    metadata['ext_dict']['yMin'] = yMin
    metadata['ext_dict']['yMax'] = yMax
    hdf5_file.close        
    
    return reflArray, metadata, wavelengths

def calc_clip_index(clipExtent, h5Extent, xscale=1, yscale=1):
    
    '''calc_clip_index calculates the indices relative to a full flight line extent of a subset given a clip extent in UTM m (x,y)
    --------
    Parameters
    --------
        clipExtent: dictionary of extent of region 
        h5Extent: dictionary of extent of h5 file (from the h5refl2array function, this corresponds to metadata['ext_dict'])
        xscale: optional, pixel size in the x-dimension, default is 1m (applicable to NEON reflectance data)
        yscale: optional, pixel size in the y-dimension, default is 1m (applicable to NEON reflectance data)
    --------
    Returns 
        newRaster.tif: geotif raster created from reflectance array and associated metadata
    --------
    Notes
    --------
    The clipExtent must lie within the extent of the h5Extent for this function to work. 
    If clipExtent exceets h5Extent in any direction, the function will return an error message. 
    --------
    Example:
    --------
    clipExtent = {'xMax': 368100.0, 'xMin': 367400.0, 'yMax': 4306350.0, 'yMin': 4305750.0}
    calc_clip_index(clipExtent, sercRefl, xscale=1, yscale=1) ''' 
    
    h5rows = h5Extent['yMax'] - h5Extent['yMin']
    #h5cols = h5Extent['xMax'] - h5Extent['xMin']    
    
    ind_ext = {}
    ind_ext['xMin'] = round((clipExtent['xMin']-h5Extent['xMin'])/xscale)
    ind_ext['xMax'] = round((clipExtent['xMax']-h5Extent['xMin'])/xscale)
    ind_ext['yMax'] = round(h5rows - (clipExtent['yMin']-h5Extent['yMin'])/xscale)
    ind_ext['yMin'] = round(h5rows - (clipExtent['yMax']-h5Extent['yMin'])/yscale)
    
    return ind_ext

def subset_clean_band(reflArray,reflArray_metadata,clipIndex,bandIndex):
    
    '''subset_clean_band extracts a band from a reflectance array, subsets it to the specified clipIndex, and applies the no data value and scale factor
    --------
    Parameters
    --------
        reflArray: reflectance array of dimensions (y,x,426) from which multiple bands (typically 3) are extracted
        reflArray_metadata: reflectance metadata associated with reflectance array (generated by h5refl2array function)
        clipIndex: ditionary; indices relative to a full flight line extent of a subset given a clip extent (generated by calc_clip_index function)
        bandIndex: band number to be extracted (integer between 1-426)
    --------
    Returns 
        bandCleaned: array of subsetted band with no data value set to NaN and scale factor applied
    --------
    See Also
    --------
    h5refl2array: 
        reads in a NEON hdf5 reflectance file and returns the reflectance array, select metadata, and the wavelength dataset
    calc_clip_index:
        calculates the indices relative to a full flight line extent of a subset given a clip extent in UTM m (x,y)
    --------
    Example:
    --------
    sercRefl, sercRefl_md, wavelengths = h5refl2array('NEON_D02_SERC_DP1_20160807_160559_reflectance.h5')
    clipExtent = {'xMax': 368100.0, 'xMin': 367400.0, 'yMax': 4306350.0, 'yMin': 4305750.0}
    serc_subInd = calc_clip_index(clipExtent,sercRefl_md['ext_dict']) 
    
    serc_b58_subset = subset_clean_band(sercRefl,sercRefl_md,serc_subInd,58) '''
    
    bandCleaned = reflArray[clipIndex['yMin']:clipIndex['yMax'],clipIndex['xMin']:clipIndex['xMax'],bandIndex-1].astype(np.float)
    bandCleaned[bandCleaned==int(reflArray_metadata['noDataVal'])]=np.nan
    bandCleaned = bandCleaned/reflArray_metadata['scaleFactor']
    
    return bandCleaned 

#########################################################################
#########################################################################
def spectral_sign_workflow_single(filename):
    #start timer
    print("Time at start: ")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Read in file: ", filename)
    start_time = time.time()
    
    # Extract reflectance array, the metadata, and the wavelengths
    print("\nTime: %s seconds" % (time.time() - start_time))
    
    print("Getting extent...\n")
    time1 = time.time()
    siteRefl, siteRefl_md, wavelengths = h5refl2array(inPath+filename)
    
    # Define the extent in a dictionary
    extDict = {}
    extDict['xMin'] = siteRefl_md['ext_dict']['xMin'] 
    extDict['xMax'] = siteRefl_md['ext_dict']['xMax'] 
    extDict['yMin'] = siteRefl_md['ext_dict']['yMin'] 
    extDict['yMax'] = siteRefl_md['ext_dict']['yMax'] 
    print(filename, " extent: xMin = ", extDict['xMin'], ", xMax = ", extDict['xMax'], ", yMin = ", extDict['yMin'], ", yMax = ", extDict['yMax'], 
          "\nTime: %s seconds" % (time.time() - time1))
    
    print("\nCleaning the tile...")
    time2 = time.time()
    
    # Remove the non-values and apply the scale factor
    #View and apply scale factor and data ignore value 
    site_clip = calc_clip_index(extDict,siteRefl_md['ext_dict']) 
    
    # Setup blank array to be filled
    site_clean = np.zeros((1000,1000))    
    
    # Clean data for only the bands of interest 
    for i in bands_of_interest:
        site_intermediate = subset_clean_band(siteRefl,siteRefl_md,site_clip,i)
        if i == 0:
            site_clean = site_intermediate
        else:
            site_clean = np.dstack([site_clean,site_intermediate]) #change to np.stack

    
    print("\nTime: %s seconds" % (time.time() - time2))
    
    print("\n\nTime at start: \n")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("\n\nPulling spectral data...")
    time3 = time.time()  
    
    # Isolate the extent of tile
    x_extent = int(filename[18:24])
    y_extent = int(filename[25:32])
    
    # Subset the data based on NLCD type and if it's within the tile's extent that's currently read in
    subset_by_extent_DF = DF_training.loc[(DF_training['X_Ext'] == x_extent) & (DF_training['Y_Ext'] == y_extent)]
    subset_by_extent_EF = EF_training.loc[(EF_training['X_Ext'] == x_extent) & (EF_training['Y_Ext'] == y_extent)]
    subset_by_extent_PH = PH_training.loc[(PH_training['X_Ext'] == x_extent) & (PH_training['Y_Ext'] == y_extent)]
    subset_by_extent_SS = SS_training.loc[(SS_training['X_Ext'] == x_extent) & (SS_training['Y_Ext'] == y_extent)]
    subset_by_extent_OW = OW_training.loc[(OW_training['X_Ext'] == x_extent) & (OW_training['Y_Ext'] == y_extent)]
    subset_by_extent_DHI = DHI_training.loc[(DHI_training['X_Ext'] == x_extent) & (DHI_training['Y_Ext'] == y_extent)]
    subset_by_extent_DF_CC = DF_CC_training.loc[(DF_training['X_Ext'] == x_extent) & (DF_CC_training['Y_Ext'] == y_extent)]
#    subset_by_extent_PH_EX = PH_EX_training.loc[(PH_training['X_Ext'] == x_extent) & (PH_training['Y_Ext'] == y_extent)]

    subset_by_extents_DF = subset_by_extent_DF.copy()
    subset_by_extents_EF = subset_by_extent_EF.copy()
    subset_by_extents_PH = subset_by_extent_PH.copy()
    subset_by_extents_SS = subset_by_extent_SS.copy()
    subset_by_extents_OW = subset_by_extent_OW.copy()
    subset_by_extents_DHI = subset_by_extent_DHI.copy()
    subset_by_extents_DF_CC = subset_by_extent_DF_CC.copy()
#    subset_by_extents_PH_EX = subset_by_extent_PH_EX.copy()
   
    # Extract data that's within the extent and then pull by x/y location
    def test_get_value(df):
        Index = 0
        time5 = time.time()
        for i in df.index:
            x_ind = df.at[i, 'X_Ind']
            y_ind = df.at[i, 'Y_Ind']
            for j in range(0, len(bandColNames)):
                try:
                    df .at[i, bandColNames[j] ] = site_clean[y_ind, x_ind, j] 
                except:
                    pass
            if Index % 10000 == 0: 
                print ("/nWorking on row: ", Index)
                print ("MDT Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
                print ("Time from start: \nTime: %s seconds" % (time.time() - time5))
            Index +=1
    
    # Print dimensions and pull values for each NLCD type        
    print("Dimensions of DF subset: ", subset_by_extents_DF.shape)        
    test_get_value(subset_by_extents_DF)
    
    print("Dimensions of EF subset: ", subset_by_extents_EF.shape)        
    test_get_value(subset_by_extents_EF)
    
    print("Dimensions of PH subset: ", subset_by_extents_PH.shape)        
    test_get_value(subset_by_extents_PH)
    
    print("Dimensions of SS subset: ", subset_by_extents_SS.shape)        
    test_get_value(subset_by_extents_SS)
    
    print("Dimensions of OW subset: ", subset_by_extents_OW.shape)        
    test_get_value(subset_by_extents_OW)
    
    print("Dimensions of DHI subset: ", subset_by_extents_DHI.shape)        
    test_get_value(subset_by_extents_DHI)
    
    print("Dimensions of DF_CC subset: ", subset_by_extents_DF_CC.shape)        
    test_get_value(subset_by_extents_DF_CC)
    
#    print("Dimensions of PH subset: ", subset_by_extents_PH_EX.shape)        
#    test_get_value(subset_by_extents_PH_EX)

    print("\nTime at end: \n")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("\nSpectral data pulled: \nTime: %s seconds" % (time.time() - time3))
    
    print("\nWriting DF spectral file")
    DF_training2 = subset_by_extents_DF.dropna()
    DF_training2.to_csv(workspacePath + "/DF_Training/DF_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting EF spectral file")
    EF_training2 = subset_by_extents_EF.dropna()
    EF_training2.to_csv(workspacePath + "/EF_Training/EF_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting PH spectral file")
    PH_training2 = subset_by_extents_PH.dropna()
    PH_training2.to_csv(workspacePath + "/PH_Training/PH_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting SS spectral file")
    SS_training2 = subset_by_extents_SS.dropna()
    SS_training2.to_csv(workspacePath + "/SS_Training/SS_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting OW spectral file")
    OW_training2 = subset_by_extents_OW.dropna()
    OW_training2.to_csv(workspacePath + "/OW_Training/OW_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting DHI spectral file")
    DHI_training2 = subset_by_extents_DHI.dropna()
    DHI_training2.to_csv(workspacePath + "/DHI_Training/DHI_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting DF_CC spectral file")
    DF_CC_training2 = subset_by_extents_DF_CC.dropna()
    DF_CC_training2.to_csv(workspacePath + "/DF_CC_Training/DF_CC_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
#    print("\nWriting PH spectral file")
#    PH_EX_training2 = subset_by_extents_PH_EX.dropna()
#    PH_EX_training2.to_csv(workspacePath + "/PH_Training/PH_EX_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nTotal Time: %s seconds" % (time.time() - start_time)) 


def spectral_sign_workflow_training(filename):

    #start timer
    print("Time at start: ")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Read in file: ", filename)
    start_time = time.time()
    
    # Extract reflectance array, the metadata, and the wavelengths
    print("\nTime: %s seconds" % (time.time() - start_time))
    
    print("Getting extent...\n")
    time1 = time.time()
    siteRefl, siteRefl_md, wavelengths = h5refl2array(inPath+filename)
    
    # Define the extent in a dictionary
    extDict = {}
    extDict['xMin'] = siteRefl_md['ext_dict']['xMin'] 
    extDict['xMax'] = siteRefl_md['ext_dict']['xMax'] 
    extDict['yMin'] = siteRefl_md['ext_dict']['yMin'] 
    extDict['yMax'] = siteRefl_md['ext_dict']['yMax'] 
    print(filename, " extent: xMin = ", extDict['xMin'], ", xMax = ", extDict['xMax'], ", yMin = ", extDict['yMin'], ", yMax = ", extDict['yMax'], 
          "\nTime: %s seconds" % (time.time() - time1))
    
    print("\nCleaning the tile...")
    time2 = time.time()
    
    # Remove the non-values and apply the scale factor
    #View and apply scale factor and data ignore value 
    site_clip = calc_clip_index(extDict,siteRefl_md['ext_dict']) 
    
    # Setup blank array to be filled
    site_clean = np.zeros((1000,1000))    
    
    # Clean data for only the bands of interest 
    for i in bands_of_interest:
        site_intermediate = subset_clean_band(siteRefl,siteRefl_md,site_clip,i)
        if i == 0:
            site_clean = site_intermediate
        else:
            site_clean = np.dstack([site_clean,site_intermediate]) #change to np.stack

    
    print("\nTime: %s seconds" % (time.time() - time2))
    
    print("\nTime at start: \n")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("\nPulling spectral data...")
    time3 = time.time()  
    
    # Isolate the extent of tile
    x_extent = int(filename[18:24])
    y_extent = int(filename[25:32])
    
    # Subset the data based on NLCD type and if it's within the tile's extent that's currently read in
    subset_by_extent_DF = DF_training.loc[(DF_training['X_Ext'] == x_extent) & (DF_training['Y_Ext'] == y_extent)]
    subset_by_extent_EF = DF_training.loc[(EF_training['X_Ext'] == x_extent) & (EF_training['Y_Ext'] == y_extent)]
    subset_by_extent_PH = DF_training.loc[(PH_training['X_Ext'] == x_extent) & (PH_training['Y_Ext'] == y_extent)]
    subset_by_extent_SS = DF_training.loc[(SS_training['X_Ext'] == x_extent) & (SS_training['Y_Ext'] == y_extent)]
    subset_by_extent_OW = DF_training.loc[(OW_training['X_Ext'] == x_extent) & (OW_training['Y_Ext'] == y_extent)]
    subset_by_extent_DHI = DF_training.loc[(DHI_training['X_Ext'] == x_extent) & (DHI_training['Y_Ext'] == y_extent)]
    subset_by_extent_DF_CC = DF_CC_training.loc[(DF_training['X_Ext'] == x_extent) & (DF_CC_training['Y_Ext'] == y_extent)]

    subset_by_extents_DF = subset_by_extent_DF.copy()
    subset_by_extents_EF = subset_by_extent_EF.copy()
    subset_by_extents_PH = subset_by_extent_PH.copy()
    subset_by_extents_SS = subset_by_extent_SS.copy()
    subset_by_extents_OW = subset_by_extent_OW.copy()
    subset_by_extents_DHI = subset_by_extent_DHI.copy()
    subset_by_extents_DF_CC = subset_by_extent_DF_CC.copy()
    
    # Extract data that's within the extent and then pull by x/y location
    def test_get_value(df):
        Index = 0
        #time5 = time.time()
        for i in df.index:
            x_ind = df.at[i, 'X_Ind']
            y_ind = df.at[i, 'Y_Ind']
            for j in range(0, len(bandColNames)):
                try:
                    df .at[i, bandColNames[j] ] = site_clean[y_ind, x_ind, j] 
                except:
                    pass
            if Index % 10000 == 0: 
                print ("Working on row: ", Index)
                #print ("MDT Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
                #print ("Time from start: \nTime: %s seconds" % (time.time() - time5))
            Index +=1
    
    # Print dimensions and pull values for each NLCD type        
    print("Dimensions of DF subset: ", subset_by_extents_DF.shape)        
    test_get_value(subset_by_extents_DF)
    
    print("Dimensions of EF subset: ", subset_by_extents_EF.shape)        
    test_get_value(subset_by_extents_EF)
    
    print("Dimensions of PH subset: ", subset_by_extents_PH.shape)        
    test_get_value(subset_by_extents_PH)
    
    print("Dimensions of SS subset: ", subset_by_extents_SS.shape)        
    test_get_value(subset_by_extents_SS)
    
    print("Dimensions of OW subset: ", subset_by_extents_OW.shape)        
    test_get_value(subset_by_extents_OW)
    
    print("Dimensions of DHI subset: ", subset_by_extents_DHI.shape)        
    test_get_value(subset_by_extents_DHI)
    
    print("Dimensions of DF_CC subset: ", subset_by_extents_DF_CC.shape)        
    test_get_value(subset_by_extents_DF_CC)

    print("\n\nTime at end: \n")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("\nSpectral data pulled: \nTime: %s seconds" % (time.time() - time3))
    
    print("\nWriting DF spectral file")
    
    DF_training2 = subset_by_extents_DF.dropna()
    DF_training2.to_csv(workspacePath + "/DF_Training/DF_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting EF spectral file")
    EF_training2 = subset_by_extents_EF.dropna()
    EF_training2.to_csv(workspacePath + "/EF_Training/EF_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting PH spectral file")
    PH_training2 = subset_by_extents_PH.dropna()
    PH_training2.to_csv(workspacePath + "/PH_Training/PH_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting SS spectral file")
    SS_training2 = subset_by_extents_SS.dropna()
    SS_training2.to_csv(workspacePath + "/SS_Training/SS_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting OW spectral file")
    OW_training2 = subset_by_extents_OW.dropna()
    OW_training2.to_csv(workspacePath + "/OW_Training/OW_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting DHI spectral file")
    DHI_training2 = subset_by_extents_DHI.dropna()
    DHI_training2.to_csv(workspacePath + "/DHI_Training/DHI_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nWriting DF_CC spectral file")
    DF_CC_training2 = subset_by_extents_DF_CC.dropna()
    DF_CC_training2.to_csv(workspacePath + "/DF_CC_Training/DF_CC_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    
    print("\nTotal Time: %s seconds" % (time.time() - start_time)) 
    
##########################################################################
##########################################################################
    
def spectral_sign_workflow_full_site(filename):

    #start timer
    print("Time at start: ")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Read in file: ", filename)
    start_time = time.time()
    
    # Extract reflectance array, the metadata, and the wavelengths
    print("\nTime: %s seconds" % (time.time() - start_time))
    
    print("Getting extent...\n")
    time1 = time.time()
    siteRefl, siteRefl_md, wavelengths = h5refl2array(inPath+filename)
    
    # Define the extent in a dictionary
    extDict = {}
    extDict['xMin'] = siteRefl_md['ext_dict']['xMin'] 
    extDict['xMax'] = siteRefl_md['ext_dict']['xMax'] 
    extDict['yMin'] = siteRefl_md['ext_dict']['yMin'] 
    extDict['yMax'] = siteRefl_md['ext_dict']['yMax'] 
    print(filename, " extent: xMin = ", extDict['xMin'], ", xMax = ", extDict['xMax'], ", yMin = ", extDict['yMin'], ", yMax = ", extDict['yMax'], 
          "\nTime: %s seconds" % (time.time() - time1))
    
    print("\nCleaning the tile...")
    time2 = time.time()
    
    # Remove the non-values and apply the scale factor
    #View and apply scale factor and data ignore value 
    site_clip = calc_clip_index(extDict,siteRefl_md['ext_dict']) 
    
    #site_clean = subset_clean_refl(siteRefl, siteRefl_md, site_clip)
    site_clean = np.zeros((1000,1000))
    
    
    site_intermediate = subset_clean_band(siteRefl,siteRefl_md,site_clip,24)
    site_clean = site_intermediate
    
    for i in bands_of_interest:
        site_intermediate = subset_clean_band(siteRefl,siteRefl_md,site_clip,i)
        if i == 0:
            site_clean = site_intermediate
        else:
            site_clean = np.dstack([site_clean,site_intermediate]) #change to np.stack

    
    print("\nTime: %s seconds" % (time.time() - time2))
    
    print("\nTime at start: \n")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("\nPulling spectral data...")
    time3 = time.time()  
    
    # Isolate the extent of tile
    x_extent = int(filename[18:24])
    y_extent = int(filename[25:32])
    
    subset_by_extent = fs_clip.loc[(fs_clip['X_Ext'] == x_extent) & (fs_clip['Y_Ext'] == y_extent)]

    subset_by_extents = subset_by_extent.copy()
    
    print("Dimensions of subset: ", subset_by_extents.shape)
 
    # Extract data that's within the extent and then pull by x/y location
    def test_get_value(df):
        Index = 0
        #time5 = time.time()
        for i in df.index:
            x_ind = df.at[i, 'X_Ind']
            y_ind = df.at[i, 'Y_Ind']
            for j in range(0, len(bandColNames)):
                try:
                    df .at[i, bandColNames[j] ] = site_clean[y_ind, x_ind, j] 
                except:
                    pass
            if Index % 10000 == 0: 
                print ("Working on row: ", Index)
                #print ("MDT Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
                #print ("Time from start: \nTime: %s seconds" % (time.time() - time5))
            Index +=1
            
    test_get_value(subset_by_extents)

    print("\nTime at end: ")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("\nSpectral data pulled: \nTime: %s seconds" % (time.time() - time3))
    
    print("\nExporting .csv ...")
    fs_clip2 = subset_by_extents.dropna()
    fs_clip2.to_csv(fs_workspacePath + "Full_Site_" + filename[18:24] + "_" + filename[25:32] + ".csv")
    print("\nTotal Time: %s seconds" % (time.time() - start_time)) 
    
###############################################################################
###############################################################################
    
# How to run for all the tiles for training data
for file in os.listdir(inPath):
    if file.endswith('.h5'):
        spectral_sign_workflow_single(file)


## Merge all csvs into one
all_df = glob.glob(os.path.join(workspacePath, "DF_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
df_total = (pd.read_csv(f) for f in all_df)
concatenated_df   = pd.concat(df_total, ignore_index=True)

all_ef = glob.glob(os.path.join(workspacePath, "EF_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
ef_total = (pd.read_csv(f) for f in all_ef)
concatenated_ef   = pd.concat(ef_total, ignore_index=True)

all_dfcc = glob.glob(os.path.join(workspacePath, "DF_CC_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
dfcc_total = (pd.read_csv(f) for f in all_dfcc)
concatenated_dfcc   = pd.concat(dfcc_total, ignore_index=True)

all_ss = glob.glob(os.path.join(workspacePath, "SS_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
ss_total = (pd.read_csv(f) for f in all_df)
concatenated_ss   = pd.concat(ss_total, ignore_index=True)

all_dhi = glob.glob(os.path.join(workspacePath, "DHI_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
dhi_total = (pd.read_csv(f) for f in all_dhi)
concatenated_dhi   = pd.concat(dhi_total, ignore_index=True)

all_ow = glob.glob(os.path.join(workspacePath, "OW_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
ow_total = (pd.read_csv(f) for f in all_ow)
concatenated_ow   = pd.concat(ow_total, ignore_index=True)

all_ph = glob.glob(os.path.join(workspacePath, "PH_Training/","*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
ph_total = (pd.read_csv(f) for f in all_ph)
concatenated_ph   = pd.concat(ph_total, ignore_index=True)
       
# Remove NA values from spectral data
DF_training = concatenated_df.dropna()
EF_training = concatenated_ef.dropna()
DHI_training = concatenated_dhi.dropna()
SS_training = concatenated_ss.dropna()
PH_training = concatenated_ph.dropna()
OW_training = concatenated_ow.dropna()
DF_CC_training = concatenated_dfcc.dropna()        

# Drop extra columns
DF_training = DF_training.drop(columns = ['POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y','X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'])
EF_training = EF_training.drop(['POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y',
                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'], axis = 1)
DHI_training = DHI_training.drop(['POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y',
                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'], axis = 1)
OW_training = OW_training.drop(['POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y',
                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'], axis = 1)
PH_training = PH_training.drop(['POINT_X', 'POINT_Y', 'LOWERL_X', 'lowerL_Y',
                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'], axis = 1)
SS_training = SS_training.drop(['POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y',
                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'], axis = 1)
DF_CC_training = DF_CC_training.drop(['POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y',
                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext', 'Y_Ind_Flip'], axis = 1)
#PH_EX_training2 = PH_EX_training.drop(columns = ['Unnamed: 0', 'GRID_Code', 'POINT_X', 'POINT_Y', 'LOWERL_X', 'LOWERL_Y',
#                  'X_Ind', 'Y_Ind', 'X_Ext', 'Y_Ext'])


        
# Remove NA values from spectral data
#DF_training2 = DF_training.dropna()
#EF_training2 = EF_training.dropna()
#OW_training2 = OW_training.dropna()
#DHI_training2 = DHI_training.dropna()
#SS_training2 = SS_training.dropna()
#PH_training2 = PH_training.dropna()
#DF_CC_training2 = DF_CC_training.dropna()
#PH_EX_training2 = PH_EX_training.dropna()

# Export training data to .csvs
DF_training.to_csv(csvPath + "DF_training_spectral.csv")
EF_training.to_csv(csvPath + "EF_training_spectral.csv")
OW_training.to_csv(csvPath + "OW_training_spectral.csv")
DHI_training.to_csv(csvPath + "DHI_training_spectral.csv")
SS_training.to_csv(csvPath + "SS_training_spectral.csv")
PH_training.to_csv(csvPath + "PH_training_spectral.csv")
DF_CC_training.to_csv(csvPath + "DF_CC_training_spectral.csv")
#PH_EX_training2.to_csv(csvPath + "PH_EX_training_spectral.csv")

###############################################################################
################# SECTION 2: FIT TO MACHINE LEARNING MODEL ####################
###############################################################################
# If starting from this point, read in training data
DF_training2 = pd.read_csv(csvPath + "DF_training_spectral.csv")
EF_training2 = pd.read_csv(csvPath + "EF_training_spectral.csv")
PH_training2 = pd.read_csv(csvPath + "PH_training_spectral.csv")
SS_training2 = pd.read_csv(csvPath + "SS_training_spectral.csv")
OW_training2 = pd.read_csv(csvPath + "OW_training_spectral.csv")
DF_CC_training2 = pd.read_csv(csvPath + "DF_CC_training_spectral.csv")
DHI_training2 = pd.read_csv(csvPath + "DHI_training_spectral.csv")

DF_training = DF_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])
EF_training = EF_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])
PH_training = PH_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])
SS_training = SS_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])
OW_training = OW_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])
DF_CC_training = DF_CC_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])
DHI_training = DHI_training2.drop(columns = ['Unnamed: 0', 'GRID_CODE'])


# Combine .csvs
nlcd_train = pd.concat([DF_training, PH_training, SS_training, OW_training, DHI_training, EF_training, DF_CC_training])
nlcd_train.to_csv(workspacePath + "nlcd_train.csv")

# Can pick up at this point if session ended
#nlcd_train = pd.read_csv(workspacePath + "nlcd_train_v2.csv")

# Features names in training data
lidar_name = ["LiDAR_HGT"]
names = lidar_name + bandColNames 

print (len(nlcd_train[nlcd_train['LiDAR_HGT'] == 0]))
nlcd_train2 = nlcd_train.replace(to_replace = 0, value = 0.000001)

# Separate into data vs column name
nlcd_x_train = pd.DataFrame(nlcd_train2.loc[:, 'LiDAR_HGT':'B416'])
nlcd_y_train = pd.DataFrame(nlcd_train2['NLCD'])

# Split into train and test data randomly
# This initial split is to separate out the test data to keep it preserved
x_train, x_test_full, y_train, y_test_full = train_test_split(
    nlcd_x_train, nlcd_y_train, test_size = 0.3)

# Split into train and test data randomly
# Double split so that you can use the intermediate test data to check the number of features necessary
def choose_best_bands(x_train, y_train):
    # Split data into train and test data
    x_train_minor, x_test_minor, y_train_minor, y_test_minor = train_test_split(
            x_train, y_train, test_size = 0.3)

    # Convert dataframes to arrays
    # Turn dataframes into arrays so they can fit in the model
    y_train_minor_array = np.array(y_train_minor)
    y_train_minor_array_flatten = np.ravel(y_train_minor_array)
    
    # Perform RFE (recursive feature elimination) to remove the number of features being processed
    # Move through the number of features if it passes the accuracy, confusion matrix, and f1 score tests
    # Number of features = [361, 32, 26, 21, 17, 13, 10, 7, 5]
    select = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=5)
    
    # Time how long it takes to fit the model
    print("Time at start of RFE fitting: ")
    print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    start_time = time.time()
    
    # Fit the model
    select.fit(x_train_minor, y_train_minor_array_flatten)
    
    #print(select)
    
    # Print the time it took to fit the model
    print("Time: %s seconds" % (time.time() - start_time))
    
    # Transform the training and test data to include just the bands that were selected to stay
    x_train_rfe = select.transform(x_train_minor)
    x_test_rfe = select.transform(x_test_minor)
    
    # Check the score of the rfe model
    accuracy = RandomForestClassifier().fit(x_train_rfe, y_train_minor_array_flatten).score(x_test_rfe, y_test_minor)
    print("Test accuracy score: {:.3f}".format(accuracy))

    # Convert the boolean array into a list of column names
    best_bands = list(compress(names, (select.support_)))
    return best_bands

print("\nWorking on CASE 1...")
tl1 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 2...")
tl2 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 3...")
tl3 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 4...")
tl4 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 5...")
tl5 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 6...")
tl6 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 7...")
tl7 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 8...")
tl8 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 9...")
tl9 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 10...")
tl10 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 11...")
tl11 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 12...")
tl12 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 13...")
tl13 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 14...")
tl14 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 15...")
tl15 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 16...")
tl16 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 17...")
tl17 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 18...")
tl18 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 19...")
tl19 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 20...")
tl20 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 21...")
tl21 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 22...")
tl22 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 23...")
tl23 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 24...")
tl24 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 25...")
tl25 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 26...")
tl26 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 27...")
tl27 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 28...")
tl28 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 29...")
tl29 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 30...")
tl30 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 31...")
tl31 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 32...")
tl32 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 33...")
tl33 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 34...")
tl34 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 35...")
tl35 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 36...")
tl36 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 37...")
tl37 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 38...")
tl38 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 39...")
tl39 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 40...")
tl40 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 41...")
tl41 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 42...")
tl42 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 43...")
tl43 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 44...")
tl44 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 45...")
tl45 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 46...")
tl46 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 47...")
tl47 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 48...")
tl48 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 49...")
tl49 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 50...")
tl50 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 51...")
tl51 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 52...")
tl52 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 53...")
tl53 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 54...")
tl54 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 55...")
tl55 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 56...")
tl56 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 57...")
tl57 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 58...")
tl58 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 59...")
tl59 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 60...")
tl60 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 61...")
tl61 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 62...")
tl62 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 63...")
tl63 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 64...")
tl64 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 65...")
tl65 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 66...")
tl66 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 67...")
tl67 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 68...")
tl68 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 69...")
tl69 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 70...")
tl70 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 71...")
tl71 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 72...")
tl72 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 73...")
tl73 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 74...")
tl74 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 75...")
tl75 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 76...")
tl76 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 77...")
tl77 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 78...")
tl78 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 79...")
tl79 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 80...")
tl80 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 81...")
tl81 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 82...")
tl82 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 83...")
tl83 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 84...")
tl84 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 85...")
tl85 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 86...")
tl86 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 87...")
tl87 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 88...")
tl88 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 89...")
tl89 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 90...")
tl90 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 91...")
tl91 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 92...")
tl92 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 93...")
tl93 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 94...")
tl94 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 95...")
tl95 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 96...")
tl96 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 97...")
tl97 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 98...")
tl98 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 99...")
tl99 = choose_best_bands(x_train, y_train)

print("\nWorking on CASE 100...")
tl100 = choose_best_bands(x_train, y_train)

important_features = np.empty(shape = (100,5))
# Combine lists of most important features
important_features = np.vstack((tl1, tl2, tl3, tl4, tl5, tl6, tl7, tl8, tl9, tl10, tl11, tl12, tl13, tl14, tl15, tl16, tl17, tl18, tl19, tl20,
                                tl21, tl22, tl23, tl24, tl25, tl26, tl27, tl28, tl29, tl30, tl31, tl32, tl33, tl34, tl35, tl36, tl37, tl38, tl39, tl40,
                                tl41, tl42, tl43, tl44, tl45, tl46, tl47, tl48, tl49, tl50, tl51, tl52, tl53, tl54, tl55, tl56, tl57, tl58, tl59, tl60,
                                tl61, tl62, tl63, tl64, tl65, tl66, tl67, tl68, tl69, tl70, tl71, tl72, tl73, tl74, tl75, tl76, tl77, tl78, tl79, tl80,
                                tl81, tl82, tl83, tl84, tl85, tl86, tl87, tl88, tl89, tl90, tl91, tl92, tl93, tl94, tl95, tl96, tl97, tl98, tl99, tl100))


# Count number of occurences in each
unique, counts = np.unique(important_features, return_counts = True)
dict(zip(unique, counts))
imp_ft = {'unique': unique, 'counts': counts}
imp_df = pd.DataFrame(imp_ft)

# Separate out list based on bands occuring in five or more iterations
imp_temp = imp_df.loc[imp_df['counts'] >= 15]

band_temp = imp_temp['unique']

true_list = band_temp.tolist()

true_list_sorted = sorted(true_list, reverse = True)

true_list_df = pd.DataFrame(true_list_sorted)
true_list_df.to_csv(workspacePath + "RFC_LiDAR_Adj_Run100_Choose5_Bands.csv")

#true_list = ['LiDAR_HGT', 'B14', 'B13', 'B17', 'B18', 'B11', 'B15', 'B264', 'B265', 'B266', 'B318', 'B4', 'B416', 'B5', 'B60', 'B62']

# Try intersecting lists
#intersection = list(set(tl0) & set(tl1) & set(tl2) & set(tl3) & set(tl4))
#print(intersection) 

# Subset the columns from the original nlcd dataframes
x_train_small = x_train[true_list_sorted]
x_test_small = x_test_full[true_list_sorted]

# Set parameters for classification
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num = 10)]
max_depth = [3, 5, 7, 10, 15, 20, 30, 50]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Set up Randomized Search using RFC
rfc = RandomForestClassifier()
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 5, verbose = 2, 
                                random_state = 42, n_jobs = -1)

# Convert dataframes to arrays
# Turn dataframes into arrays so they can fit in the model
x_train_array = np.array(x_train_small)
y_train_array = np.array(y_train)
y_train_array_flatten = np.ravel(y_train_array)

# Select new random subset

# Time how long it takes to fit the model
print("\nApply model to test data")
print("Time at start of RFC fitting: ")
print ("MDT: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
time1 = time.time()

# Fit model to varying parameters
rfc_random.fit(x_train_array, y_train_array_flatten)

# Print the time it took to fit the model
print("\nTime: %s seconds" % (time.time() - time1))

# Set prediction for test values
prediction = rfc_random.predict(x_test_small)

prediction_probability = rfc_random.predict_proba(x_test_small)
print(prediction)
print(prediction_probability)

# Visual check to see if results "make sense"
y_test_array = np.array(y_test_full)
y_test_array_flatten = np.ravel(y_test_array)
#print(y_test_array_flatten)

# Confusion Matrix
confusion = confusion_matrix(y_test_array_flatten, prediction)
print("Confusion Matrix:\n{}:".format(confusion))

precision, recall, fscore, support = score(y_test_array_flatten, prediction)

fscore_avg = np.average(fscore)
print("Fscore average: ", fscore_avg)

# Save model to disk for later
joblib_file = (workspacePath + "RFC_LiDAR_Adj_Run100_Choose5_20181101.pkl")
joblib.dump(rfc_random, joblib_file)

#list(true_list)

###############################################################################
################# SECTION 3: PULL AND CLEAN FULL SITE DATA ####################
###############################################################################

# Repeat data pulling steps for full site

# Names of .csvs to read in
# Csv includes x and y extents, x and y indicies, and LiDAR canopy height information
full_site_csv = "fullsite_clipped_lidar.csv"

# Create dataframes showing the location of the training data points
Full_Site_loc = pd.read_csv(workspacePath+full_site_csv) 
#column_names = list(Full_Site_loc)
#print(column_names)

#Full_Site_loc.head()
Full_Site_loc['LiDAR_HGT'] = Full_Site_loc['RASTERVALU']
Full_Site_loc = Full_Site_loc.drop(['Field1', 'RASTERVALU'],axis = 1)

# Copy dataframes
fs_clip = Full_Site_loc.copy()

print( len(fs_clip[fs_clip['LiDAR_HGT'] == 0]))

fs_clip2 = fs_clip.replace(to_replace = 0, value = 0.000001)


# How to run for all the tiles for full site
for file in os.listdir(inPath):
    if file.endswith('.h5'):
        spectral_sign_workflow_full_site(file)
        
# Read in produced .csvs and merge them
                    # use your path
print("Read in all spectral .csvs")
all_files = glob.glob(os.path.join(fs_workspacePath, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

full_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_full   = pd.concat(full_from_each_file, ignore_index=True)

#Read in subset of bands to extract
print("Read in optimal bands .csv")
bands = pd.read_csv(workspacePath + "RFC_LiDAR_Adj_Run20_Choose100_Bands.csv")
bands.columns = ['id', 'bands']
band_list = bands['bands'].tolist()
subset_of_bands = ['GridCode', 'X_Ext', 'Y_Ext', 'X_Loc', 'Y_Loc', 'X_Ind', 'Y_Ind'] + band_list

print("Subset full site spectral based on optimal bands")
concat_clean = pd.DataFrame(concatenated_full.loc[:, subset_of_bands])

# Merge dataframes with LiDAR and with Spectral Info
#spectral_data = pd.DataFrame(concat_clean.loc[:, band_list])
#list(concat_clean)
#concat_clean['LiDAR_HGT'].head()
#print("Merging spectral .csvs")
#
#merged_df = fs_clip.assign(**concat_clean)
#       
## Remove NA values from spectral data
#fs_clip2 = merged_df.dropna()
#
#list(fs_clip2)
#
## Isolate bands selected earlier to go into the machine learning algorithm
#
## Add empty band columns
#true_list_length = (len(true_list_sorted)-1)*(-1)
#true_list_bands = true_list_sorted[true_list_length:]
#
#for i in true_list_bands:
#    fs_clip[i]=""
#
#subset_of_bands = ['GridCode', 'X_Ext', 'Y_Ext', 'X_Loc', 'Y_Loc', 'X_Ind', 'Y_Ind'] + true_list_bands
#
#subset_df =fs_clip2[subset_of_bands]

# Export training data to .csvs
#fs_clip2.to_csv(outPath + "Full_Site_Spectral_v3.csv")

###############################################################################
################# SECTION 4: APPLY ML ALGORITHM TO FULL SITE ##################
###############################################################################
#fs_clip2 = pd.read_csv(outPath + "Full_Site_Spectral2.csv")
# Separate full site into training data and name
print("Separate out feature data")
x_full = pd.DataFrame(concat_clean.loc[:, 'LiDAR_HGT':])

print("Read in model")
# Name of job file
joblib_file = (workspacePath + "RFC_LiDAR_Adj_Run20_Choose100_20181101.pkl")
# Read in model if need be
rfc_random1 = joblib.load(joblib_file)

print("Applying model to full site")

# Predict the full site
full_predict = rfc_random1.predict(x_full)
full_predict_df = pd.DataFrame(full_predict)
full_predict_df.columns = ['Predicted']

# Write to csv
#full_predict_df.to_csv(outPath + "Full_Site_Predicted.csv")

# Merge the predicted NLCD class to the df with spatial information
print("Merging predictions to dataframe")
full_predict_merge = pd.concat([concat_clean, full_predict_df], axis = 1, sort = False)

# Remove extra columns
list(full_predict_merge2)
full_predict_merge2 = full_predict_merge.drop(columns = list(full_predict_merge)[7:53])

#full_predict_merge.head()

print("Exporting full site NLCD predictions")

full_predict_merge2.to_csv(outPath+"Full_Site_Merged_run20_choose100.csv")

#full_predict_merge2['index'][full_predict_merge2['X_Loc'] == None]

#full_predict_merge2 = full_predict_merge
#full_predict_merge_test = full_predict_merge2[(full_predict_merge2['X_Loc'] == 744469)]

###############################################################################
################ SECTION 5: CONVERT FULL SITE TO 30X30M PIXELS ################
###############################################################################
full_predict_merge2 = pd.read_csv(outPath + "Full_Site_Merged_run20_choose100.csv")
list(full_predict_merge2)
full_predict_merge2 = full_predict_merge2.drop(['Unnamed: 0'], axis = 1)

############## WORKSPACE ##################
#a = full_predict_merge2.dropna(axis=1)
###########################################

# Check value counts
#print("Show counts of each land cover type on 1x1m scale")
#print(full_predict_merge2['Predicted'].value_counts())

# Create function to assign NLCD value based on percent coverage

# Pull out series
#full_predict = full_predict_merge2['NLCD']
#nlcd_pct = NLCD_perc # to test

def assign_NLCD_test(nlcd_pct):
    new_nlcd_df = pd.Series.to_frame(nlcd_pct)
    new_nlcd_df['NLCD_name'] = new_nlcd_df.index
    try:
        df = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'DF', 'Predicted'][0]
    except:
        df = 0.0
    try:
        dfcc = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'DF_CC', 'Predicted'][0]
    except:
        dfcc = 0.0
    try:
        ef = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'EF', 'Predicted'][0]
    except:
        ef = 0.0
    try:
        ph = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'PH', 'Predicted'][0]
    except:
        ph = 0.0
    try:
        ss = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'SS', 'Predicted'][0] 
    except:
        ss = 0.0
    try:
        ow = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'OW', 'Predicted'][0]
    except:
        ow = 0.0
    try:
        dhi = new_nlcd_df.loc[new_nlcd_df['NLCD_name'] == 'DHI', 'Predicted'][0]
    except:
        dhi = 0.0
    
    # Round to nearest ten-thousandth place for ease of printing
    rdf = round(df, 3)
    rdfcc = round(dfcc, 3)
    ref = round(ef, 3)
    rph = round(ph, 3)
    rss = round(ss, 3)
    row = round(ow, 3)
    rdhi = round(dhi, 3)
    
    if (df + dfcc >= 75):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'DF'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is pure DF")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (ef >= 75):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'EF'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is pure EF")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (((df+dfcc+ef) > 20) & (((df+dfcc/(df+ef+dfcc))*100)>=75)):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'DF'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is mixed DF")  
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (((df+ef+dfcc) > 20) & (((ef/(df+ef+dfcc))*100)>=75)):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'EF'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is mixed EF") 
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (((df+ef+dfcc) > 20) & (((df+dfcc/(df+ef+dfcc))*100)<75) & (((ef/(df+ef+dfcc))*100)<75)):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'MF'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is MF")  
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (ph >= 50):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'PH'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is PH")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (ss >= 50):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'SS'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is SS")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (ow >= 75):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'OW'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is OW")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (dhi >= 80):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'DHI'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is DHI")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif ((ss > 30) & (ss > ph)):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'SS'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is weak SS")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif ((ph > 30) & (ph > ss)):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'PH'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is weak PH")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (dhi > 30):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'DHI'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is weak DHI")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    elif (ow > 30):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'OW'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is weak OW")
        #print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)    
    elif ((df + dfcc+ef+ph+ss+ow+dhi < 95) & (df + dfcc+ef+ph+ss+ow+dhi > 0)):
        NLCD_for_30m_pixel.at[i, 'X_Loc'] = thirty_pixel_list[i][0]
        NLCD_for_30m_pixel.at[i,'Y_Loc'] = thirty_pixel_list[i][1]
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'NA'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is partial and therefore not defined")
        print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)
    else:
        NLCD_for_30m_pixel.at[i,'NLCD'] = 'NA2'
        print("Pixel X: ", thirty_pixel_list[i][0], "and Y: ", thirty_pixel_list[i][1], "is wonky. CHECK IMMEDIATELY")
        print("This pixel has DF:", rdf, "DF_CC: ", rdfcc, "EF: ", ref, "PH: ", rph, "SS: ", rss, "OW: ", row, "DHI: ", rdhi)

# Starting and ending coordinates for SCBI
#x_start = 744453
#x_end = 749943
#y_start = 4305476
#y_end = 4310096

# Subset of site where known gaps exist
x_start = 746253
x_end = 747153
y_start = 4307276
y_end = 4308176


# List of 30 integers since the pixels will be 30x30m
steps = list(range(30))
#print(steps)

# Creates an int indicating the start of each pixel
x_step = list(range(x_start, x_end, 30))
y_step = list(range(y_start, y_end, 30))

# Read predicted NLCD types for each 1x1m pixel from the ML algorithm
# Only need if starting from this step
#full_predict = pd.read_csv(outPath + 'Full_Site_Predicted.csv')

#reindex
print("Reindexing")
x=full_predict_merge2.reset_index()
full_predict_merge2=x.set_index(['X_Loc','Y_Loc']).sort_index()

list(full_predict_merge2)

# 1. Create lists of xs and ys to slice out of full site
thirty_pixel_list = list(itertools.product(x_step, y_step))

# For each tuple in thiry_pixel_list get a 30x30 list of pair values
#[thirty_pixel_list[0][0]+s for s in steps]

#Initiate final data frame

# Create new blank dataframe
NLCD_for_30m_pixel = pd.DataFrame(columns = ['X_Loc', 'Y_Loc', 'NLCD'])

#i=2  #To test
# Get a list of X_loc plus each value step. Repeat for as many steps as there are
for i in range(0,len(thirty_pixel_list)):
    print("Getting x list for ", i)
    x_list = list(itertools.chain.from_iterable(itertools.repeat([thirty_pixel_list[i][0]+s for s in steps], len(steps))))
    print("Getting y list for ", i)
    y_list = list(itertools.chain.from_iterable(zip(*itertools.repeat([thirty_pixel_list[i][1]+s for s in steps], len(steps)))))

    print("Zip x and y lists together")
    xys = list(zip(x_list, y_list))
    
    print("Slice values if data exists")
    # 2. Slice values out of full site if data exists
    try:
        #newDF=full_predict_merge2.drop(['GridCode', 'X_Ext', 'Y_Ext'], axis = 1).loc[xys]
        #newDF=newDF.drop(['index'], axis=1)
        #newDF=newDF.drop(['X_Loc', 'Y_Loc'], axis=1)
        #NLCD_sum = newDF.groupby(['Predicted']).Predicted.count()
        #NLCD_perc = (newDF.groupby(['Predicted']).Predicted.count() / 9)
        coords = ['x_list', 'y_list']
        newDF = pd.DataFrame(full_predict_merge2.loc[:, coords])
    
        assign_NLCD_test(NLCD_perc)
    except:
        pass
    #Summarize



# Add centroid coordinates for arcGIS
NLCD_for_30m_pixel['X_Centroid'] = NLCD_for_30m_pixel['X_Loc'] + 15
NLCD_for_30m_pixel['Y_Centroid'] = NLCD_for_30m_pixel['Y_Loc'] + 15

print("Exporting NLCD csv")
NLCD_for_30m_pixel.to_csv(outPath+"NLCD_30m_Centroid_run100_choose4_v2.csv")
print("New NLCD has been exported")
    #Add back to final dataframe with 
#    X_Loc=thirty_pixel_list[i][0]
#    Y_Loc=thirty_pixel_list[i][1]
    #Add summary nlcd class













