import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr
import fiona
import geopandas
import rasterio
import numpy as np
import random
from shapely.geometry import Point
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from numpy import arange

import sampling
import extract
# import vaildation
import prepare_data
import model

# import vaildation
# adjust **training_ratio**
def split_traning_validation(in_df, column_name="TraVa", training_ratio = 0.666, random_state=1 ,verbose=True) :

    total_feature_count = in_df.shape[0]
    if verbose : print("total records : {}".format(total_feature_count))

    training_count = int(total_feature_count*training_ratio)

    out_df=in_df.copy(deep=True)
    out_df[column_name]=0

    training_df=out_df.sample(n=training_count,random_state=random_state)
    training_df[column_name]=1

    out_df.update(training_df)

    return out_df

# 放入樹種點位
Tree_point = gpd.read_file("df01_extract.shp")

# 填寫背景與主體比例
#Tree_df = sampling.balance_target_and_background(Tree_point, background_ratio=1, verbose= False)
#data_name = 'RD_{}.shp'.format(i)
#diff_in_df.to_file(data_name)
#ex_point = extract.extracting(data_name,'elevation01.tif','aspect_r01.tif','slope01.tif','curvature01.tif','curvature_planform01.tif','curvature_profile01.tif','tsi2_01.tif','tsi3_01.tif','tsi4_01.tif','tsi5_01.tif','tsi2_10_01.tif','tsi3_10_01.tif','tsi4_10_01.tif','tsi5_10_01.tif', 'TP1_01.tif', 'TP2_01.tif', 'TP3_01.tif')
#save_Tree_df = ex_point.to_file(data_name)

# 調整訓練與測試欄位
# It can change 
training_ratio = list(np.arange(0.4, 0.95, 0.05))

Tree_va=[]
for i in training_ratio:
    diff_training_ratio = split_traning_validation(Tree_point, column_name="TraVa", training_ratio = i, random_state=1 ,verbose=True)
    Tree_va.append(diff_training_ratio)

#選擇變數
Tree_pp= []
for i in Tree_va:
    pre_data = prepare_data.prepare_data(pd.DataFrame(i.drop(columns='geometry')), x_indexes= [4, 8, 9, 10 ,11 ,12, 13 ,14, 15, 16, 17], y_index = [3])
    Tree_pp.append(pre_data)

long = list(arange(1 , len(training_ratio)+1 , 1))

###run model
Tree_RF = []
for i in long:
    run_model = model.RF_entropy(Tree_pp[i-1], n_estimators=500, max_depth=7, min_samples_split=40, min_samples_leaf=15)
    Tree_RF.append(run_model)

#print result
print(Tree_RF)

###
y_acc_train = (Tree_RF[0])[:1] + (Tree_RF[1])[:1] + (Tree_RF[2])[:1] + (Tree_RF[3])[:1] + (Tree_RF[4])[:1]+ (Tree_RF[5])[:1]+ (Tree_RF[6])[:1]+ (Tree_RF[7])[:1]+ (Tree_RF[8])[:1]+ (Tree_RF[9])[:1]+ (Tree_RF[10])[:1]
y_acc_test = (Tree_RF[0])[1:2] + (Tree_RF[1])[1:2] + (Tree_RF[2])[1:2] + (Tree_RF[3])[1:2] + (Tree_RF[4])[1:2] + (Tree_RF[5])[1:2] + (Tree_RF[6])[1:2] + (Tree_RF[7])[1:2]+ (Tree_RF[8])[1:2]+ (Tree_RF[9])[1:2]+ (Tree_RF[10])[1:2]
y_kappa_train = (Tree_RF[0])[2:3] + (Tree_RF[1])[2:3] + (Tree_RF[2])[2:3] + (Tree_RF[3])[2:3] + (Tree_RF[4])[2:3] + (Tree_RF[5])[2:3] + (Tree_RF[6])[2:3] + (Tree_RF[7])[2:3]+ (Tree_RF[8])[2:3]+ (Tree_RF[9])[2:3]+ (Tree_RF[10])[2:3]
y_kappa_test = (Tree_RF[0])[3:4] + (Tree_RF[1])[3:4] + (Tree_RF[2])[3:4] + (Tree_RF[3])[3:4] + (Tree_RF[4])[3:4] + (Tree_RF[5])[3:4] + (Tree_RF[6])[3:4]+ (Tree_RF[7])[3:4]+ (Tree_RF[8])[3:4]+ (Tree_RF[9])[3:4]+ (Tree_RF[10])[3:4]

training_ratio = ['0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75' , '0.80' , '0.85', '0.9']
plt.xlabel('training_ratio') 
plt.ylabel('accuracy') 
plt.title('Adjust the ratio of training data')      
plt.ylim(0.6, 1.0)
plt.plot(training_ratio, y_acc_train, color='Navy', marker='o', linestyle = '--', label='acc_train')
plt.plot(training_ratio, y_acc_test, color = 'DarkSlateBlue', marker='o', linestyle = '-', label='acc_test')
plt.plot(training_ratio, y_kappa_train, color = 'ForestGreen', marker='o', linestyle = '--', label='kappa_train')
plt.plot(training_ratio, y_kappa_test, color = 'Olive', marker='o', linestyle = '-', label='kappa_test')
plt.legend(loc = 'lower left')

plt.show()

