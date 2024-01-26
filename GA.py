# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:47:59 2023

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
from MMGEN import *

def g1g2_encoder(br):
    if (br == 0):
        bits = np.array([0,0])
    elif (br == 1):
        bits = np.array([0,1])
    elif (br == 5):
        bits = np.array([1,0])
    else:
        print("Error")
    return(bits)

def g1g2_decoder(bits):
    if ((bits == np.array([0,0])).all()):
        br = 0
    elif ((bits == np.array([0,1])).all()):
        br = 1
    elif ((bits == np.array([1,0])).all()):
        br = 5
    else:
        print("Error ", bits)
    return(br)

def bin2deci(binary_num):
    str_num = ''.join(map(str,binary_num))
    dec_num = int(str_num, 2)
    return dec_num

def deci2bin(dec_num, dig):
    binary_num= format(dec_num, '0'+str(dig)+'b')
    binary = np.array(list(binary_num)).astype(int)
    return binary

def g3g4_decoder(bits):
    pc1 = bin2deci(bits[0:7])
    pc2 = bin2deci(bits[7:11])
    return(pc1+(pc2/10))

def g3g4_encoder(pc):
  g3 = deci2bin(int(pc),7)
  g4 = deci2bin(round((pc%1)*10),4)
  return(np.concatenate((g3,g4), 0))

def gene_to_real(chrm):
  real = np.array([g1g2_decoder(chrm[0:2]), g1g2_decoder(chrm[2:4]),
                        g3g4_decoder(chrm[4:15])])
  return(real)


def real_to_gene(real):
  chrm = np.concatenate((g1g2_encoder(real[0]), g1g2_encoder(real[1]), g3g4_encoder(real[2])))
  return(chrm)

def dna_to_data(dna):
  data = np.apply_along_axis(gene_to_real, 1, dna)
  return(data)

def repair_legality(chrm):
  chrm_rep = chrm.copy()
  #chrm = chrm
  if ((chrm[0:4] == np.array([0,0,0,0])).all()):
    #All 2 brands cant be 0
    chrm_rep = initialize_pop(1,False)[0]
  elif ((chrm[0:2] == np.array([1,1])).all()):
    #Either of the brand can be [1,1]
    chrm_rep = initialize_pop(1,False)[0]
  elif ((chrm[2:4] == np.array([1,1])).all()):
      chrm_rep = initialize_pop(1,False)[0]
  elif ((chrm[0:2] == np.array([0,0])).all() and g3g4_decoder(chrm[4:15])!=0):
    #If pno_br = 0, pc should be 0
    chrm_rep[4:15] = g3g4_encoder(0)
  elif ((chrm[2:4] == np.array([0,0])).all() and g3g4_decoder(chrm[4:15])!=100):
    #If mzo_br = 0, pc should be 100
    chrm_rep[4:15] = g3g4_encoder(100)
  elif g3g4_decoder(chrm[4:15]) > 100:
    chrm_rep[4:15] = g3g4_encoder(round(random.uniform(0, 99.9), 1))
  elif (g3g4_decoder(chrm[4:15]) == 100 and (chrm[2:4] == np.array([0,0])).all()!=True):
    chrm_rep[2:4] = np.array([0,0])
  elif (g3g4_decoder(chrm[4:15]) == 0 and (chrm[0:2] == np.array([0,0])).all()!=True):
    chrm_rep[0:2] = np.array([0,0])
  else:
    chrm_rep = chrm.copy()
    #print("No: Repair", chrm_rep)
    return (chrm_rep)
  #print("    Repair",chrm_rep)
  return (repair_legality(chrm_rep))

def initialize_pop(size, first_run):
  population_bit = np.zeros((size,15)).astype(int)
  if first_run == True:
    population_bit[0,] = real_to_gene(np.array([1,0,100]))
    population_bit[1,] = real_to_gene(np.array([0,1,0]))
    population_bit[2,] = real_to_gene(np.array([5,0,100]))
    population_bit[3,] = real_to_gene(np.array([0,5,0]))
    population_bit[4,] = real_to_gene(np.array([1,1,10.1]))
    population_bit[5,] = real_to_gene(np.array([1,1,20.2]))
    population_bit[6,] = real_to_gene(np.array([1,5,30.3]))
    population_bit[7,] = real_to_gene(np.array([1,5,40.4]))
    population_bit[8,] = real_to_gene(np.array([5,1,50.5]))
    population_bit[9,] = real_to_gene(np.array([5,1,60.6]))
    population_bit[10,] = real_to_gene(np.array([5,5,70.7]))
    population_bit[11,] = real_to_gene(np.array([5,5,80.8]))
    population_bit[12,] = real_to_gene(np.array([1,1,90.9]))
    start = 13
    for i in range(start,size):
      g1 = deci2bin(random.randint(1,2),2)
      g2 = deci2bin(random.randint(1,2),2)
      pc = round(random.uniform(0.1, 99.9), 1)
      g3g4 = g3g4_encoder(pc)
      chrm = np.concatenate((g1,g2,g3g4))
      population_bit[i,] = repair_legality(chrm)
  else:
    start = 0
    for i in range(start,size):
      g1 = deci2bin(random.randint(0,2),2)
      g2 = deci2bin(random.randint(0,2),2)
      pc = round(random.uniform(0, 100), 1)
      g3g4 = g3g4_encoder(pc)
      chrm = np.concatenate((g1,g2,g3g4))
      population_bit[i,] = repair_legality(chrm)
  return(population_bit)

# def mutate (chrm):
#   #inverses one random genome in the chromosome
#   chance = round(random.uniform(0, 1),2)
#   if (chance < 0.25):
#       ind = random.randint(0,2)
#   elif (chance >= 0.25 and chance <0.5):
#       ind = random.randint(2,4)
#   elif (chance >= 0.5 and chance <0.75):
#       ind = random.randint(4,11)
#   else:
#     ind = random.randint(11,14)
#   chrm[ind] = 1 if chrm[ind]==0 else 0
#   return(chrm)

def mutate (chrm):
  #inverses one random genome in the chromosome
  ind = random.randint(0,14)
  chrm[ind] = 1 if chrm[ind]==0 else 0
  return(chrm)

def crossover (chrm1, chrm2):
  chance = round(random.uniform(0, 1),2)
  if (chance < 0.20):
      offspring = np.append(chrm1[0:2], chrm2[2:15])
  elif (chance >= 0.20 and chance <0.40):
      offspring = np.append(chrm1[0:4], chrm2[4:15])
  elif (chance >= 0.40 and chance <0.60):
      ind = random.randint(4,11)
      offspring = np.append(chrm1[0:ind], chrm2[ind:15])
  elif (chance >= 0.60 and chance <0.80):
      ind = random.randint(11,14)
      offspring = np.append(chrm1[0:ind], chrm2[ind:15])
  else:
    ind = random.randint(0,14)
    offspring = np.append(chrm1[0:ind], chrm2[ind:15])
  return(offspring)


def breed(parent1_ind, parent2_ind, population):
  chrm1 = population[parent1_ind,0:15]
  chrm2 = population[parent2_ind,0:15]
  ch = round(random.uniform(0, 1),2)
  if(ch>0.05):
    offspring=crossover(chrm1,chrm2)
  else:
    offspring =mutate(chrm1)
  rep_offspring = repair_legality(offspring)
  return(rep_offspring)

def breed_pop(Knowns, Knowns_meta, test, test_label,population):
  fitn_population = fitness_function(Knowns, Knowns_meta,test,test_label, population)
  pop = fitn_population[:,0:15].astype(int)
  new_pop_np = np.zeros((population.shape[0],15)).astype(int)
  j = 0
  #only top 10% of the list allowed the breed.
  length = math.floor(population.shape[0]/10)
  new_pop_np[j,] = pop[0,0:15].copy()
  new_pop_np[(j+1),] = pop[1,0:15].copy()
  new_pop_np[(j+2),] = pop[2,0:15].copy()
  new_pop_np[(j+3),] = pop[3,0:15].copy()
  new_pop_np[(j+4),] = pop[4,0:15].copy()
  j = j+4
  for i in range(0,(length-1)):
    aa = breed(i,(i+1),pop)
    #print("a", aa.copy())
    b = breed((i+1),i,pop)
    #print("b", b)
    c = breed((length-1-i),i,pop)
    #print("c", c)
    d = breed(i,(length-1-i),pop)
    #print("d", d)

    new_pop_np[(j+1),] = aa.copy()
    new_pop_np[(j+2),] = b.copy()
    new_pop_np[(j+3),] = c.copy()
    new_pop_np[(j+4),] = d.copy()
    #print(new_pop_np)
    j = j+4

  for jj in (range(j,population.shape[0])):
      new_pop_np[jj,] = initialize_pop(1, False)[0]
  return(new_pop_np)

def fitness_function(Knowns, Knowns_meta, test, test_label, population):
    return(fitness_function_1(Knowns, Knowns_meta, test, test_label, population))


def fitness_function_1(Knowns, Knowns_meta, test, test_label, population):
    scores = np.zeros((population.shape[0],1))
    for i in range(0,population.shape[0]):
        chrm = population[i,]
        #print("Fitness: ", chrm)
        pc = g3g4_decoder(chrm[4:15])
        pnobr = g1g2_decoder(chrm[0:2])
        mzobr = g1g2_decoder(chrm[2:4])
        gen = mmgen(Knowns, Knowns_meta, pnobr, mzobr, pc, 100-pc)
        cms = calc_cms(test, gen)
        scores[i,] = -cms
    pop = np.append(population, scores, axis = 1)
    pop = pd.DataFrame(pop).sort_values(15, ascending=False).to_numpy()
    #print(pop)
    return(pop)

def fitness_function_2(Knowns, Knowns_meta, test, test_label, population):
    scores = np.zeros((population.shape[0],1))
    for i in range(0,population.shape[0]):
        chrm = population[i,]
        #print("Fitness: ", chrm)
        pc = g3g4_decoder(chrm[4:15])
        pnobr = g1g2_decoder(chrm[0:2])
        mzobr = g1g2_decoder(chrm[2:4])
        #gen = mmgen_shift(Knowns, Knowns_meta, pnobr, mzobr, pc, 100-pc, str(test_label['spec_batch'].values[0])+"_"+str(test_label['pno_br'].values[0])+"_"+str(test_label['mzo_br'].values[0]))

        gen = mmgen_shift(Knowns, Knowns_meta, pnobr, mzobr, pc, 100-pc, str(test_label['spec_batch'].values[0])+"_"+str(pnobr)+"_"+str(mzobr))
        cms = calc_cms(test, gen)
        scores[i,] = -cms
    pop = np.append(population, scores, axis = 1)
    pop = pd.DataFrame(pop).sort_values(15, ascending=False).to_numpy()
    #print(pop)
    return(pop)

def genetic_algo (Knowns, Knowns_meta, test_spec, test_label, popsize, gensize, breakcond):

  ppopulation = initialize_pop(popsize, True)
  performance = np.zeros((gensize, 2)) #Avg and Best
  breakcond_i = 0
  for i in range(0,gensize):
    #print(i)
    ppopulation = breed_pop(Knowns, Knowns_meta, test_spec,test_label, ppopulation)
    fit_score = fitness_function(Knowns, Knowns_meta, test_spec,test_label,ppopulation)[:,-1]
    performance[i,] = np.array((sum(fit_score)/len(fit_score),max(fit_score)))

    #Stops the process if the result has not improved the last x generations
    if(performance[i,1] > max(performance[0:(i+1),1])):
      breakcond_i = 0
    else:
      breakcond_i = breakcond_i+1
    if breakcond_i >= breakcond:
        break
    else:
        continue
  #print("GT: ", test_label['all_label'].values[0], " Pred: ", gene_to_real(ppopulation[0,]))
  # print("GT: ", test_label, " Pred: ", gene_to_real(breeded_population[0,])[0],
  #       gene_to_real(breeded_population[0,])[1], "_", gene_to_real(breeded_population[0,])[2])

  return(performance, ppopulation)

def get_RA(pred, gt):
    diff = list(map(lambda x,y: abs(x-y), pred, gt))
    ra1 = sum(map(lambda x: 1 if x<=1 else 0, diff))
    ra2 = sum(map(lambda x: 1 if x<=2 else 0, diff))
    ra3 = sum(map(lambda x: 1 if x<=3 else 0, diff))
    ra4 = sum(map(lambda x: 1 if x<=4 else 0, diff))
    ra5 = sum(map(lambda x: 1 if x<=5 else 0, diff))
    ra10 = sum(map(lambda x: 1 if x<=10 else 0, diff))
    ra20 = sum(map(lambda x: 1 if x<=20 else 0, diff))
    return(ra1, ra2, ra3, ra4, ra5, ra10, ra20)
