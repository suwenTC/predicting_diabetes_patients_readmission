import math
import numpy as np

def compute_performance_measures(tn, fp, fn, tp):
    with np.errstate(divide='ignore'):
        sen = 0 if tp+fn == 0 else (1.0*tp)/(tp+fn)
    
    with np.errstate(divide='ignore'):
        spc = 0 if tn+fp == 0 else (1.0*tn)/(tn+fp)
    
    with np.errstate(divide='ignore'):
        f1s = 0 if (2.0*tp+fn+fp)==0 else (2.0*tp)/(2.0*tp+fn+fp)
        
    with np.errstate(divide='ignore'):
        ppr = 0 if (tp+fp)==0 else (1.0*tp)/(tp+fp)
    
    with np.errstate(divide='ignore'):
        npr = 0 if (tn+fn)==0 else (1.0*tn)/(tn+fn)
    
    with np.errstate(divide='ignore'):
        acc = 0 if tp+fp+tn+fn==0 else (tp+tn)/(tp+fp+tn+fn)
        
    with np.errstate(invalid='ignore'):
        didx = math.log(1+acc, 2)+math.log(1+(sen+spc)/2, 2)
        
    return (sen, spc, f1s, ppr, npr)