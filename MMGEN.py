# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:50:23 2023

@author: alagl
"""

import os
import pandas as pd
import numpy as np
import random
import math
import timeit
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
import warnings
import pickle


def mmgen(Knowns,Knowns_meta, b1, b2, p1, p2):
    if (b1 == 0):
        r2 = Knowns.loc[(Knowns_meta['pno_purity'] == 0) &
                        (Knowns_meta['mzo_br'] == b2) ]
        gen_spec = r2.sample(n=6, replace=True)
        return (gen_spec)
    elif (b2 == 0):
        r1 = Knowns.loc[(Knowns_meta['pno_purity'] == 100) & (Knowns_meta['pno_br'] == b1) ]
        gen_spec = r1.sample(n=6, replace=True)
        return (gen_spec)
    else:
        r1 = Knowns.loc[(Knowns_meta['pno_purity'] == 100) &
                        (Knowns_meta['pno_br'] == b1) ]
        r2 = Knowns.loc[(Knowns_meta['pno_purity'] == 0) &
                        (Knowns_meta['mzo_br'] == b2) ]
        purity_mat = np.array([[100,0],[0,100]])
        target_mat = np.array([[p1],[p2]])
        ratio_mix_mat = np.linalg.solve(purity_mat, target_mat)
        gen_spec_a = (r1.sample(n=6, replace=True)).mul(ratio_mix_mat[0][0])
        gen_spec_b = (r2.sample(n=6, replace=True)).mul(ratio_mix_mat[1][0])
        gen_spec = gen_spec_a+gen_spec_b.values
        #gen_spec = pd.DataFrame(gen_spec_a.to_numpy()+gen_spec_b.to_numpy())
        return (gen_spec)

def calc_cms(test, gen):
    #Wrapper to call different CMS
    return(calc_cms_canberra(test, gen))

def calc_cms_canberra(test, gen):
    #return(calc_cms_euc(test, gen))
    a1 = abs(test - gen.median(0))
    a2 = abs(test) + abs(gen.median(0))
    a3 = a1 / a2
    sc = sum(a3.values.tolist()[0])
    return(sc)

def calc_cms_man(test, gen):
    #return(calc_cms_euc(test, gen))
    a1 = abs(test - gen.median(0))
    sc = sum(a1.values.tolist()[0])
    return(sc)

def calc_cms_euc(test, gen):
    a = test - gen.median(0)
    aa = a.pow(2)
    return(math.sqrt(sum(list(aa.values)[0])))

def calc_cms_sqeuc(test, gen):
    a = test - gen.median(0)
    aa = a.pow(2)
    return(sum(list(aa.values)[0]))

def calc_cms_pc(test, gen,n=5):
    x = pd.concat([gen,test])
    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(x)
    gen_pcs = principalComponents[:-1]
    test_pcs = principalComponents[-1]
    a1 = abs(test_pcs - np.median(gen_pcs,0))
    a2 = abs(test_pcs) + abs(np.median(gen_pcs,0))
    a3 = a1 / a2
    sc = sum(a3)
    return(sc)

def load_params(batch_bb):
    wdir = "C:/Users/alagl/OneDrive - Wilmar International Limited/Desktop/IPP Research/Projects/GA"
    f1 = pd.read_csv (wdir+"/Data_Py/ShiftParams/"+batch_bb+"_m0.csv")
    m0 = f1.drop('Unnamed: 0', axis=1)
    f2 = pd.read_csv (wdir+"/Data_Py/ShiftParams/"+batch_bb+"_a0.csv")
    a0 = f2.drop('Unnamed: 0', axis=1)
    a0 = a0.transpose()
    f3 = pd.read_csv (wdir+"/Data_Py/ShiftParams/"+batch_bb+"_m100.csv")
    m100 = f3.drop('Unnamed: 0', axis=1)
    f4 = pd.read_csv (wdir+"/Data_Py/ShiftParams/"+batch_bb+"_a100.csv")
    a100 = f4.drop('Unnamed: 0', axis=1)
    a100 = a100.transpose()
    return m0,a0,m100,a100

def make_dict():
    batches = ["20200901", "20200902", "20200903", "20200904", "20200907", "20200908",
                "20200909", "20200910", "20200911", "20200914", "20200915", "20200916",
                "20200917", "20201123", "20201124", "20201125", "20201126", "20201127",
                "20201201", "20201202", "20201203", "20201204", "20201207", "20210111",
                "20210112"]
    dicts = {}
    for i in batches:
      for j in ["1","5"]:
            a1 = i+"_"+j+"_m0"
            a2 = i+"_"+j+"_a0"
            a3 = i+"_"+j+"_m100"
            a4 = i+"_"+j+"_a100"
            dicts[a1], dicts[a2], dicts[a3], dicts[a4] = load_params(i+"_"+j)
    return dicts

def mmgen_shift(Knowns,Knowns_meta, b1, b2, p1, p2, batch):
    #m0,a0,m100,a100 = load_params(batch)
    if (batch[0:8] == "20200831"):
            return(mmgen(Knowns, Knowns_meta, b1, b2, p1, p2))
    else:
            # print(batch)
            # m0 = dicts[batch[0:9]+batch[11]+"_m0"]
            # m100 = dicts[batch[0:10]+"_m100"]
            # a0 = dicts[batch[0:9]+batch[11]+"_a0"]
            # a100 = dicts[batch[0:10]+"_a100"]
            if (b1 == 0):
                r2 = Knowns.loc[(Knowns_meta['pno_purity'] == 0) &
                                (Knowns_meta['mzo_br'] == b2) ]
                gen_spec = r2.sample(n=6, replace=True)
                if (batch[11] != '0'):
                        m0 = dicts[batch[0:9]+batch[11]+"_m0"]
                        a0 = dicts[batch[0:9]+batch[11]+"_a0"]
                        m0.index = gen_spec.columns
                        gen_spec = gen_spec.dot(m0) + a0.values
                        gen_spec.columns = Knowns.columns
                return (gen_spec)
            elif (b2 == 0):
                r1 = Knowns.loc[(Knowns_meta['pno_purity'] == 100) & (Knowns_meta['pno_br'] == b1) ]
                gen_spec = r1.sample(n=6, replace=True)
                if (batch[9] != '0'):
                        m100 = dicts[batch[0:10]+"_m100"]
                        a100 = dicts[batch[0:10]+"_a100"]
                        m100.index = gen_spec.columns
                        gen_spec = gen_spec.dot(m100) + a100.values
                        gen_spec.columns = Knowns.columns
                return (gen_spec)
            else:
                r1 = Knowns.loc[(Knowns_meta['pno_purity'] == 100) &
                                (Knowns_meta['pno_br'] == b1) ]
                r2 = Knowns.loc[(Knowns_meta['pno_purity'] == 0) &
                                (Knowns_meta['mzo_br'] == b2) ]
                purity_mat = np.array([[100,0],[0,100]])
                target_mat = np.array([[p1],[p2]])
                ratio_mix_mat = np.linalg.solve(purity_mat, target_mat)
                gen_spec_a = (r1.sample(n=6, replace=True))
                gen_spec_b = (r2.sample(n=6, replace=True))

                m100 = dicts[batch[0:10]+"_m100"]
                a100 = dicts[batch[0:10]+"_a100"]
                m100.index = gen_spec_a.columns
                gen_spec_a = gen_spec_a.dot(m100)+a100.values
                gen_spec_a = gen_spec_a.mul(ratio_mix_mat[0][0])
                m0 = dicts[batch[0:9]+batch[11]+"_m0"]
                a0 = dicts[batch[0:9]+batch[11]+"_a0"]
                m0.index = gen_spec_b.columns
                gen_spec_b = gen_spec_b.dot(m0)+a0.values
                gen_spec_b = gen_spec_b.mul(ratio_mix_mat[1][0])

                # if (batch[9] != '0'):
                #         m100 = dicts[batch[0:10]+"_m100"]
                #         a100 = dicts[batch[0:10]+"_a100"]
                #         m100.index = gen_spec_a.columns
                #         gen_spec_a = gen_spec_a.dot(m100)+a100.values
                #         gen_spec_a = gen_spec_a.mul(ratio_mix_mat[0][0])

                # if (batch[11] != '0'):
                #         m0 = dicts[batch[0:9]+batch[11]+"_m0"]
                #         a0 = dicts[batch[0:9]+batch[11]+"_a0"]
                #         m0.index = gen_spec_b.columns
                #         gen_spec_b = gen_spec_b.dot(m0)+a0.values
                #         gen_spec_b = gen_spec_b.mul(ratio_mix_mat[1][0])

                gen_spec = gen_spec_a+gen_spec_b.values
                #gen_spec = pd.DataFrame(gen_spec_a.to_numpy()+gen_spec_b.to_numpy())
                gen_spec.columns = Knowns.columns
                return (gen_spec)

fname = "C:/Users/alagl/OneDrive - Wilmar International Limited/Desktop/IPP Research/Projects/GA/Data_Py/ShiftParams/shift_dict.pickle"
if (os.path.isfile(fname)):
        with open(fname, 'rb') as f:
                dicts = pickle.load(f)
else:
        dicts  = make_dict()
        with open(fname, 'wb') as f:
                pickle.dump(dicts, f, pickle.HIGHEST_PROTOCOL)


# def load_params(batch_bb):
#     wdir = "C:/Users/alagl/OneDrive - Wilmar International Limited/Desktop/IPP Research/Projects/GA"
#     f1 = pd.read_csv (wdir+"/Data_Py/Purity/"+batch_bb+"_m0.csv")
#     m0 = f1.drop('Unnamed: 0', axis=1)
#     f2 = pd.read_csv (wdir+"/Data_Py/Purity/"+batch_bb+"_a0.csv")
#     a0 = f2.drop('Unnamed: 0', axis=1)
#     a0 = a0.transpose()
#     f3 = pd.read_csv (wdir+"/Data_Py/Purity/"+batch_bb+"_m100.csv")
#     m100 = f3.drop('Unnamed: 0', axis=1)
#     f4 = pd.read_csv (wdir+"/Data_Py/Purity/"+batch_bb+"_a100.csv")
#     a100 = f4.drop('Unnamed: 0', axis=1)
#     a100 = a100.transpose()
#     return m0,a0,m100,a100

# def make_dict():
#     batches = ["20200901", "20200902", "20200903", "20200904", "20200907", "20200908",
#                "20200909", "20200910", "20200911", "20200914", "20200915", "20200916",
#                "20200917", "20201123", "20201124", "20201125", "20201126", "20201127",
#                "20201201", "20201202", "20201203", "20201204", "20201207", "20210111",
#                "20210112"]
#     dicts = {}
#     for i in batches:
#         a1 = i+"_m0"
#         a2 = i+"_a0"
#         a3 = i+"_m100"
#         a4 = i+"_a100"
#         dicts[a1], dicts[a2], dicts[a3], dicts[a4] = load_params(i)
#     return dicts

# def mmgen_shift(Knowns,Knowns_meta, b1, b2, p1, p2, batch):
#     #m0,a0,m100,a100 = load_params(batch)
#     if (batch[0:8] == "20200831"):
#             return(mmgen(Knowns, Knowns_meta, b1, b2, p1, p2))
#     else:
#             if (b1 == 0):
#                 r2 = Knowns.loc[(Knowns_meta['pno_purity'] == 0) &
#                                 (Knowns_meta['mzo_br'] == b2) ]
#                 gen_spec = r2.sample(n=6, replace=True)
#                 if (batch[11] != '0'):
#                         m0 = dicts[batch[0:8]+"_m0"]
#                         a0 = dicts[batch[0:8]+"_a0"]
#                         m0.index = gen_spec.columns
#                         gen_spec = gen_spec.dot(m0) + a0.values
#                         gen_spec.columns = Knowns.columns
#                 return (gen_spec)
#             elif (b2 == 0):
#                 r1 = Knowns.loc[(Knowns_meta['pno_purity'] == 100) & (Knowns_meta['pno_br'] == b1) ]
#                 gen_spec = r1.sample(n=6, replace=True)
#                 if (batch[9] != '0'):
#                         m100 = dicts[batch[0:8]+"_m100"]
#                         a100 = dicts[batch[0:8]+"_a100"]
#                         m100.index = gen_spec.columns
#                         gen_spec = gen_spec.dot(m100) + a100.values
#                         gen_spec.columns = Knowns.columns
#                 return (gen_spec)
#             else:
#                 r1 = Knowns.loc[(Knowns_meta['pno_purity'] == 100) &
#                                 (Knowns_meta['pno_br'] == b1) ]
#                 r2 = Knowns.loc[(Knowns_meta['pno_purity'] == 0) &
#                                 (Knowns_meta['mzo_br'] == b2) ]
#                 purity_mat = np.array([[100,0],[0,100]])
#                 target_mat = np.array([[p1],[p2]])
#                 ratio_mix_mat = np.linalg.solve(purity_mat, target_mat)
#                 gen_spec_a = (r1.sample(n=6, replace=True))
#                 gen_spec_b = (r2.sample(n=6, replace=True))

#                 m100 = dicts[batch[0:8]+"_m100"]
#                 a100 = dicts[batch[0:8]+"_a100"]
#                 m100.index = gen_spec_a.columns
#                 gen_spec_a = gen_spec_a.dot(m100)+a100.values
#                 gen_spec_a = gen_spec_a.mul(ratio_mix_mat[0][0])
#                 m0 = dicts[batch[0:8]+"_m0"]
#                 a0 = dicts[batch[0:8]+"_a0"]
#                 m0.index = gen_spec_b.columns
#                 gen_spec_b = gen_spec_b.dot(m0)+a0.values
#                 gen_spec_b = gen_spec_b.mul(ratio_mix_mat[1][0])

#                 gen_spec = gen_spec_a+gen_spec_b.values
#                 gen_spec.columns = Knowns.columns
#                 return (gen_spec)

# fname = "C:/Users/alagl/OneDrive - Wilmar International Limited/Desktop/IPP Research/Projects/GA/Data_Py/Purity/shift_dict.pickle"
# if (os.path.isfile(fname)):
#         with open(fname, 'rb') as f:
#                 dicts = pickle.load(f)
# else:
#         dicts  = make_dict()
#         with open(fname, 'wb') as f:
#                 pickle.dump(dicts, f, pickle.HIGHEST_PROTOCOL)

