from __future__ import absolute_import
from __future__ import print_function
import os

import numpy as np
import pandas as pd
import os
import re
import sys
from tensorflow import keras
import math
from sklearn import metrics
from tensorflow.keras.models import load_model


np.random.seed(7)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
modelpath = "output/"+sys.argv[1]+"/model/"

def get_all_model(modelpath):
    models = []
    for root, file_name, files in os.walk(modelpath):
        for file in files:
            models.append(os.path.join(root, file))
    return models


def ONEHOT(files):
	files = open(files, 'r')
	sample=[];ids = []
	for line in files:
		if (re.match('>',line) is None) or (not len(line.strip())): 
			value=np.zeros((201,4),dtype='float32')
			if len(line.strip()) <= 201:
				for index,base in enumerate(line.strip()):
					if re.match(base,'A|a'):
						value[index,0]=1
					if re.match(base,'T|t'):
						value[index,1]=1
					if re.match(base,'C|c'):
						value[index,2]=1
					if re.match(base,'G|g'):
						value[index,3]=1
				sample.append(value)
		elif re.match('>',line): 
			ids.append(line[1:].strip())
	files.close()
	return np.array(sample),ids

def comparison(testlabel, resultslabel):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for row1 in range(len(resultslabel)):
        if resultslabel[row1] < 0.5:
            resultslabel[row1] = 0
        else:
            resultslabel[row1] = 1
    for row2 in range(len(testlabel)):
        if testlabel[row2] == 1 and testlabel[row2] == resultslabel[row2]:
            TP = TP + 1
        if testlabel[row2] == 0 and testlabel[row2] != resultslabel[row2]:
            FP = FP + 1
        if testlabel[row2] == 0 and testlabel[row2] == resultslabel[row2]:
            TN = TN + 1
        if testlabel[row2] == 1 and testlabel[row2] != resultslabel[row2]:
            FN = FN + 1
    if TP + FN != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 0
    if TN + FP != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0
    if TP + FP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 0
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0
    if FP + TP != 0:
        FDR = FP / (FP + TP)
    else:
        FDR = 0
    if FN + TN != 0:
        FOR = FN / (FN + TN)
    else:
        FOR = 0
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 0
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 0
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN + FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = 0
    if TPR != 0 and TNR != 0:
        BM = TPR + TNR - 1
    else:
        BM = 0
    if PPV != 0 and NPV != 0:
        MK = PPV + NPV - 1
    else:
        MK = 0
    return TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK



def predict(sample_to_predict, models, label):
    print("Load %d sequences to predict!"% sample_to_predict.shape[0])
    result_array = np.zeros([sample_to_predict.shape[0]+1,1])    
    result = []
    count = 0
    f3 = open("output/"+filename2+"_result.txt",'w')
    for model in models:
        model_tf = re.split('[/-]', model)[1]
        count += 1 
        one_model = load_model(model)
        pred = one_model.predict(sample_to_predict,verbose=0)
        result.append(model_tf)
        result.append(pred)
        auc = metrics.roc_auc_score(label,pred)
        TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK = comparison(label,pred)        
        f3.writelines("TP"+"\t"+"TN"+"\t"+"FN"+"\t"+"FP"+"\t"+"TPR"+"\t"+"TNR"+"\t"+"ACC"+"\t"+"F1"+"\t"+"MCC"+"\t"+"auc"+"\n")
        f3.writelines(str(TP)+"\t"+str(TN)+"\t"+str(FN)+"\t"+str(FP)+"\t"+str(TPR)+"\t"+str(TNR)+"\t"+str(ACC)+"\t"+str(F1)+"\t"+str(MCC)+"\t"+str(auc)+"\n")
        result = np.array(result)
        result = np.vstack(result)
        result_array = np.concatenate((result_array, result), axis=1)
        keras.backend.clear_session()
        result = []       
    f3.close()
    return result_array[:,1:]






filename = "example/"+sys.argv[1]+"_test.fa"
filename2 = sys.argv[1]
f1=open("example/"+sys.argv[1]+"_label.txt",'r')
label=[]
for i in f1:
	z=i.split("\t")
	label.append(int(z[0]))

f1.close()
sample_to_predict, seq_id = ONEHOT(filename)
models = get_all_model(modelpath)
result_array = predict(sample_to_predict, models, label)

