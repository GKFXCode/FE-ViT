import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch.nn as nn
from scipy.stats import pearsonr
# from sklearn.metrics import r2_score
# from VIPL_HR_dataset import synthesize_stmap
import logging
import torchvision
import statsmodels.api as sm


def get_hr_metrics(pred, gt, string=True):
    '''
    output: Mean, Std, MAE, RMSE, MER , r
    '''
    # pred = [r[0] for r in results]
    # gt = [r[1] for r in results]
    # err = [(gt-pred) for pred, gt in results]
    err = [(p - g) for p, g in zip(pred, gt)]
    r, p_value = pearsonr(pred, gt)

    if string:
        return 'Mean: %.2f, Std: %.2f, MAE: %.2f, RMSE:%.2f, MER:%.2f %% , r:%.2f, pred_std:%.2f'% (np.mean(err), 
                                           np.std(err), 
                                           np.mean(np.abs(err)), 
                                           np.sqrt(np.mean(np.power(err,2))),
                                           np.mean(np.abs(err)/np.array(gt))*100,
                                           r,
                                           np.std(pred)
                                           )

    return (np.mean(err), 
            np.std(err), 
            np.mean(np.abs(err)), 
            np.sqrt(np.mean(np.power(err,2))),
            np.mean(np.abs(err)/np.array(gt))*100,
            r)
    
def plot_analyse_results(pred, gt):


    err = [(p-g) for p, g in zip(pred, gt)]
    r, p_value = pearsonr(pred, gt)

    fig = plt.figure(figsize=(15,15))
    fig.suptitle(get_hr_metrics(pred, gt, string=True))

    fig.subplots_adjust(top=0.93)
    
    # 误差图
    plt.subplot(3,2,1)
    plt.plot(err)
    plt.title('plot of err (gt-pred)')
    plt.legend(['err'])
    
    # 预测值与实际值图
    plt.subplot(3,2,2)
    plt.plot(gt, 'b')
    plt.plot(pred, 'r:')
    plt.title('Mean gt:%.2f, Std gt: %.2f \n Mean pred %.2f, Std pred %.2f'%(np.mean(gt), np.std(gt), np.mean(pred), np.std(pred) ))
    plt.legend(['gt', 'pred'])
    
    #误差直方图
    # plt.hist(err, bins=range(int(min(err))-1,int(max(err))+1, 5))
    # plt.xlabel('err')
    # plt.ylabel('count')
    # bland-altman plots
    ax = plt.subplot(3,2,3)

    sm.graphics.mean_diff_plot(np.array(pred), np.array(gt), ax=ax)
    
    #预测值与实际值分布图
    plt.subplot(3,2,4)
    M = max(max(pred), max(gt))
    x = [30, M]
    y = [30, M]
    plt.scatter(pred, gt, 1)
    plt.xlabel('pred')
    plt.ylabel('gt')
    plt.plot(x,y, 'r',linewidth=1)
    plt.title('min_gt:%.2f, max_gt:%.2f,\n min_pred:%.2f, max_pred:%.2f'%(min(gt), max(gt), min(pred), max(pred)))
    
    # plt.subplot(3,2,5)
    ax0 = plt.subplot2grid((9,2), (6,0), rowspan=1)
    plt.hist(err, bins=range(int(min(err))-1,int(max(err))+1, 5))
    plt.legend(['err'])
    
    ax1 = plt.subplot2grid((9,2), (7,0), rowspan=1)
    plt.hist(gt, bins=60, color='b')
    plt.legend(['gt'])
    
    ax2 = plt.subplot2grid((9,2), (8,0), rowspan=1, sharex = ax1)
    plt.hist(pred, bins=60, color='r')
    plt.legend(['pred'])

    # 预测值与实际值， 按实际值排序
    idx = [i[0] for i in sorted(enumerate(gt), key=lambda x:x[1])]
    pred = pred[idx]
    gt = gt[idx]
    plt.subplot(3,2,6)
    plt.plot(gt, 'b')
    plt.scatter(np.array(range(len(pred))), pred, 1, 'r')
    plt.legend(['gt', 'pred'])
    
    plt.show()
    return plt.gcf()