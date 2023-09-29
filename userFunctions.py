from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import cv2
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re 
import statistics
import torch
import geojson
import seaborn as sns
import os
import shutil


#Functions

def read_config_file(file_path):
    try:
        with open(file_path, 'r') as config_file:
            config_data = json.load(config_file)
        return config_data
    except FileNotFoundError:
        print(f"Error: Config file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in config file {file_path}")
        return None

def draw_annotations(row, heatmap_img, color_map, thickness = -1):
    points = row.points
    points = literal_eval(points)
    ## convert to contour
    ctr = np.array(points).reshape((-1,1,2)).astype(np.int32)
    color = color_map[row.annot_type]
    heatmap_img = cv2.drawContours(heatmap_img, [ctr], 0, color, thickness)

#Given the coordinates of a polygon, find centroid
def find_polygon_centroid(polygon_coords):
    n = len(polygon_coords)
    sum_x, sum_y = zip(*polygon_coords)
    centroid_x = sum(sum_x) / n
    centroid_y = sum(sum_y) / n
    return centroid_x, centroid_y

def tbCalc(tb_mm, percentTB):
    tbScore = 2
    
    #These values are hard coded right now. May want to consult Dr Nameeta about it
    if tb_mm >= 10 and percentTB > 3:
        tbScore = 1
    if tb_mm < 3 and percentTB < 1.5:
        tbScore = 3
        
    return tbScore

def MITcalc(mimiMITfile, hpfMITfile, celltypeFile):
    mimiMIT = pd.read_csv(mimiMITfile)
    celltype = pd.read_csv(celltypeFile)
    mitvalues = mimiMIT['annot_type'].value_counts()
    if 'MIT' not in mitvalues.index:
        mitvalues['MIT'] = 0

    MITcells = mitvalues['MIT']
    CEcells = celltype['annot_type'].value_counts()['CE']
    MITcellPerCE = round(MITcells * 10000 / CEcells, 0)
    if os.path.exists(hpfMITfile):
        hpfMIT = pd.read_csv(hpfMITfile)
        MIT_hpf = hpfMIT['annot_type'].value_counts()
    else:
        MIT_hpf = {'HPF': 0, 'MIT': 0}
        
    return MIT_hpf, MITcellPerCE

def mitoticScoreCalc(MIT_hpf):
    
    #Again hard coded values here
    lowCriteria = 7
    midCriteria = 14
    
    if MIT_hpf['MIT'] < lowCriteria:
        MITscore = 1
    elif MIT_hpf['MIT'] >= lowCriteria and MIT_hpf['MIT'] < midCriteria:
        MITscore = 2
    elif MIT_hpf['MIT'] >= midCriteria:
        MITscore = 3
    
    return MITscore

def npScoreCalc(sideMean, sideIQR, confidence):
    npScore = 2
    
    if sideMean < 8 and sideIQR < 2 and confidence < 0.25:
        npScore = 1
    
    temp = 0
    if sideMean > 9.5:
        temp += 1
    if sideIQR > 4:
        temp += 1
    if confidence > 0.4:
        temp += 1
    
    if temp > 1:
        npScore = 3
        
    return npScore

def segmentationTable(segmentationJson):
    removeSegment = 'SPA'
    segmentationData = pd.read_json(segmentationJson, orient ='index')
    segmentationData.columns = ['area in mm2']
    segmentationData['area in mm2'] = round(segmentationData['area in mm2']/1000000, 2)
    segmentationData.index = segmentationData.index.str.replace('_area', '')
    segmentationData.drop('SPA', inplace = True)
    total_count = segmentationData['area in mm2'].sum()
    segmentationData['%area'] = round((segmentationData['area in mm2'] / total_count) * 100, 1)
    segmentationDf = segmentationData.sort_values(by = '%area', ascending = False)
    return segmentationDf

def celltypeTable(celltypeFile, mimiMITfile, totalArea):
    cellTypeData = pd.read_csv(celltypeFile)
    cellTypeDf = cellTypeData['annot_type'].value_counts().reset_index()
    cellTypeDf = cellTypeDf.rename(columns = {'index': 'Segment', 0: 'Total count'})
    cellTypeDf.columns = ['Segment', 'Total count']

    mimiMIT = pd.read_csv(mimiMITfile)
    mitvalues = mimiMIT['annot_type'].value_counts()
    if 'MIT' not in mitvalues.index:
        mitvalues['MIT'] = 0
    mimi = {'Segment': 'mimi', 'Total count' : mitvalues['mimi']}
    MIT = {'Segment': 'MIT', 'Total count' : mitvalues['MIT']}
    new_row_df = pd.DataFrame([mimi])
    cellTypeDf = pd.concat([cellTypeDf, new_row_df], ignore_index=True)
    new_row_df = pd.DataFrame([MIT])
    cellTypeDf = pd.concat([cellTypeDf, new_row_df], ignore_index=True)
    cellTypeDf['per mm2'] = cellTypeDf['Total count']/totalArea
    CEvalue = cellTypeDf[cellTypeDf['Segment'] == 'CE']
    cellTypeDf['per 1000 epithelial cells'] = round((cellTypeDf['Total count'] * 1000)/CEvalue['Total count'].iloc[0], 0)
    
    return cellTypeDf

def celltypePerSegmentTable(celltypePerSegmentJson):
    celltypePerSegmentData = pd.read_json(celltypePerSegmentJson, orient ='index')
    celltypePerSegmentData.index = celltypePerSegmentData.index.str.replace('_counts_in', '')
    celltypePerSegmentData['Celltype'] = celltypePerSegmentData.index.str.split('_').str[0]
    celltypePerSegmentData['Segment'] = celltypePerSegmentData.index.str.split('_').str[1]
    celltypePerSegmentData.columns = ['value', 'Celltype', 'Segment']
    celltypeBySegmentMatrix = celltypePerSegmentData.pivot(index = 'Celltype', columns = 'Segment', values = 'value')
    celltypeBySegmentMatrix[np.isnan(celltypeBySegmentMatrix)] = 0
    celltypeBySegmentMatrix = celltypeBySegmentMatrix.drop('SPA', axis=1)  # axis=1 means we are dropping a column
    percentMatrix = celltypeBySegmentMatrix.div(celltypeBySegmentMatrix.sum(axis = 1), axis = 0)
    percentMatrix = round(percentMatrix *100, 2)

    return percentMatrix

def has_negative_coordinates(coordinates_list):
    for x, y in coordinates_list:
        if x < 0 or y < 0:
            return True
    return False

def is_centroid_inside_any_contour(centroid, contours):
    x, y = centroid
    for contour in contours:
        result = cv2.pointPolygonTest(np.array(contour), (x, y), False)
        if result > 0:
            return True
    return False