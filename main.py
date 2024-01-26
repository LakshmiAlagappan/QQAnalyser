# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:43:53 2023

@author: alagl
"""

from GA import *
from MMGEN import *

warnings.filterwarnings("ignore")
wdir = "C:/Users/alagl/OneDrive - Wilmar International Limited/Desktop/IPP Research/Projects/GA"
os.chdir(wdir)

Knowns = pd.read_csv (wdir+"/Data_Py/Knowns.csv")
Knowns = Knowns.drop('Unnamed: 0', axis=1)
Knowns_meta = pd.read_csv (wdir+"/Data_Py/Knowns_meta.csv")
Knowns_meta = Knowns_meta.drop('Unnamed: 0', axis=1)


ts1 = pd.read_csv (wdir+"/Data_Py/ts1.csv")
ts1 = ts1.drop('Unnamed: 0', axis=1)
ts1_meta = pd.read_csv (wdir+"/Data_Py/ts1_meta.csv")
ts1_meta = ts1_meta.drop('Unnamed: 0', axis=1)

ts2 = pd.read_csv (wdir+"/Data_Py/ts2.csv")
ts2 = ts2.drop('Unnamed: 0', axis=1)
ts2_meta = pd.read_csv (wdir+"/Data_Py/ts2_meta.csv")
ts2_meta = ts2_meta.drop('Unnamed: 0', axis=1)

ts3 = pd.read_csv (wdir+"/Data_Py/ts3.csv")
ts3 = ts3.drop('Unnamed: 0', axis=1)
ts3_meta = pd.read_csv (wdir+"/Data_Py/ts3_meta.csv")
ts3_meta = ts3_meta.drop('Unnamed: 0', axis=1)

ts4 = pd.read_csv (wdir+"/Data_Py/ts4.csv")
ts4 = ts4.drop('Unnamed: 0', axis=1)
ts4_meta = pd.read_csv (wdir+"/Data_Py/ts4_meta.csv")
ts4_meta = ts4_meta.drop('Unnamed: 0', axis=1)

ts5 = pd.read_csv (wdir+"/Data_Py/ts5.csv")
ts5 = ts5.drop('Unnamed: 0', axis=1)
ts5_meta = pd.read_csv (wdir+"/Data_Py/ts5_meta.csv")
ts5_meta = ts5_meta.drop('Unnamed: 0', axis=1)

tst_lst = [ts1, ts2, ts3, ts4, ts5]
tst_meta_lst = [ts1_meta, ts2_meta, ts3_meta, ts4_meta, ts5_meta]

#popsizelist = [50,100,150,200,250]
popsize = 20
#gensizelist = [50,100,150,200,250]
for pp in range(0,1):
        popsize = popsize
        gensize = 50
        cb = 10
        fitn = "Ini_D2_MMGEN_Rep"
        cmss = "Can"
        filename = wdir+"/Models_Results/bred/"+str(popsize)+"_"+str(gensize)+"_"+str(cb)+"_"+fitn+"_"+cmss+".txt"
        f = open(filename,'w')
        print("PopSize: ", popsize, file=f)
        print("GenSize: ", gensize, file=f)
        print("ContBreak: ", cb, file=f)
        print("Fitness: ", fitn, file=f)
        print("CMS: ", cmss, file=f)
        print(" ", file=f)

        for ll in range(0,5):
            tst = tst_lst[ll]
            tst_meta = tst_meta_lst[ll]

            pred_br1 = []
            pred_br2 = []
            pred_p = []
            start = timeit.default_timer()
            for j in range (0,tst.shape[0]):
                print("TS: ", ll+1, " tst: ", j+1)
                per,pop = genetic_algo(Knowns, Knowns_meta, tst.loc[[j],:],tst_meta.loc[[j], ], popsize, gensize, cb)
                pred_br1.append(gene_to_real(pop[0,])[0].astype(int))
                pred_br2.append(gene_to_real(pop[0,])[1].astype(int))
                pred_p.append(round(gene_to_real(pop[0,])[2],2))
                stop = timeit.default_timer()
                # print("Pred: ", gene_to_real(pop[0,])[0].astype(int),gene_to_real(pop[0,])[1].astype(int), "_", round(gene_to_real(pop[0,])[2],2))
                # print("Real: ", tst_meta.loc[j, 'pno_br'], tst_meta.loc[j, 'mzo_br'].astype(int), "_", tst_meta.loc[j, 'pno_purity'])

            print('TS: ', ll+1, ' Time: ', stop - start, file=f)
            a1 = list(map(lambda p, t: 1 if p == t else 0,pred_br1,tst_meta.loc[:, 'pno_br'].tolist()))
            a2 = list(map(lambda p, t: 1 if p == t else 0,pred_br2,tst_meta.loc[:, 'mzo_br'].tolist()))
            a3 = list(map(lambda p, t: 1 if p == t else 0,pred_p,tst_meta.loc[:, 'pno_purity'].tolist()))
            a4 = sum(map(lambda a,b,c: 1 if a==b==c==1 else 0,a1,a2,a3))
            print("Results of ", "TS: ", (ll+1), file=f)
            print("PNO_BR: ", sum(a1), file=f)
            print("MZO_BR: ", sum(a2), file=f)
            print("P: ", sum(a3), file=f)
            print("All: ", a4, file=f)
            print("Total: ", tst.shape[0], file=f)
            print(get_RA(pred_p,tst_meta.loc[:, 'pno_purity'].tolist()), file=f)
            print("OVER", file=f)
            print(" ", file=f)
        f.close()
