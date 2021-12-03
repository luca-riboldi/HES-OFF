# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:17:01 2021

@author: leifa
"""

#from psa_basic_mc_fast_4cycle_TEA_mod import PSA
from psa_basic_mc_fast_4cycle_TEA_EV_BD_mod import PSA
import numpy as np
import time
import PyNomad

import matplotlib.pyplot as plt
from cases import Case

#-----------------------------------------------------------
# Part I: Specify all parameters required by the script.
#-----------------------------------------------------------

#steps = 500 #2000 #
steps =100 #2000 # # I recommend to use as little steps as possible since it will speed uo the optimization 

# Define the PSA scenario that we wish to optimize: 
case_study = 'coal' #'cement' #'hydrogen' #'natural_gas_onshore' #natural_gas_offshore
case = Case(case_study)
y_feed = case.y_feed
p_feed = case.P_feed
T_feed = case.T_feed + 273.15
# if case_study == 'coal':
#     #from coal case study
#     y_feed = 0.1521
#     p_feed = 1.016
#     T_feed = 50+273.15 
# elif case_study == 'cement':
#     #from cement case study
#     y_feed = 0.247
#     p_feed = 1.01
#     T_feed = 130 + 273.15
# elif case_study == 'hydrogen':
#     # #from hydrogen case study
#     y_feed = 0.20
#     p_feed = 1.02
#     T_feed = 353.15 #298.15 #
# elif case_study == 'natural_gas_onshore':
#     #from natural gas case study
#     y_feed = 0.0432  
#     p_feed = 1.01
#     T_feed = 86.8+273.15    
# elif case_study == 'natural_gas_offshore':
#     #from natural gas case study
#     y_feed = 0.0363  
#     p_feed = 1.01
#     T_feed = 86.8+273.15 #525.8+273.15 #????
# elif case_study == 'waste':
#     #from waste case study
#     y_feed = 0.1361 
#     p_feed = 1.01
#     T_feed = 160+273.15 

# Define process costraints.
purity = 0.95 #
recovery = 0.80 #

#Define bounds
p_H_min = 1.0
p_H_max = 1.4 #
p_I_min = 0.21 # 0.11# 0.21 #0.2 + 0.01
p_I_max = 0.4 #0.7 #
p_L_min = 0.02 
p_L_max = 0.20 #0.10
T_ads_min = 298
T_ads_max = 320#360

bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

ScaleObj = [560,30]

adsorbents = ['ALPMOF_40C']


#-----------------------------------------------------------
# Part II: Specify the cost objectives and  the two constraints constraints.
#-----------------------------------------------------------

def unnormalize(x):
    return x[0]*(p_H_max-p_H_min)+p_H_min, x[1] *(p_I_max-p_I_min)+p_I_min, x[2]*(p_L_max-p_L_min)+p_L_min, x[3]*(T_ads_max-T_ads_min)+T_ads_min 

def process(x, adsorbent):
    """
    Define the adsorption process we wish to optimize. The variables are scaled to get dimentionless optimization.
    Input:
    x : an array of the 7 input variables, scaled between 0 and 1
    adsorbent: name of adsorbent
    """
    #unscale input variables
    p_H_v, p_I_v, p_L_v, T_ads_v = x  
    
    p_H = p_H_v*(p_H_max-p_H_min)+p_H_min 
    p_I = p_I_v*(p_I_max-p_I_min)+p_I_min
    p_L = p_L_v*(p_L_max-p_L_min)+p_L_min 
    T_ads = T_ads_v*(T_ads_max-T_ads_min)+T_ads_min
    
    return PSA(y_feed, p_H, p_I, p_L, p_feed, T_ads, T_feed, adsorbent, case_study, steps, use_multi_comp=True, poly_approx = False)

def bb(x):
    """Objective: Wegihted Multi-objective function 
        The current weights are scaled with the maximum value of the single 
        objective functions: 
            specific work = 560
            (working capacity = 1.4 )
            specific total mass of adsorbent = 30??
    """
    global adsorbents, ScaleObj, weights, purity, recovery 
    
    dim = x.get_n()
    X = [x.get_coord(i) for i in range(dim)]
    
    Temp = process(X,adsorbents[0])
    f = Temp[4]
    x.set_bb_output(0,f)
    g1 = purity - Temp[0]
    x.set_bb_output(1,g1)
    g2 = recovery - Temp[1]
    x.set_bb_output(2,g2)
    return 1 #np.sum(weights*[Temp[2]/560,-Temp[3]/1.4])
    #return np.sum(weights*[Temp[2]/560,-Temp[3]/1.4])
    
# ------------------------------------------------------------------------
# PART III: GP functions
# -----------------------------------------------------------------------


ub = [1,1,1,1]
lb = [0,0,0,0]
x0 = [0.393,0.5,0.136,0.1]

#params = ['BB_OUTPUT_TYPE OBJ PB PB','MAX_BB_EVAL 100','UPPER_BOUND * 1','LOWER_BOUND * 0']
#[ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = PyNomad.optimize(bb,x0,lb,ub,params)
#print ('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,f_return,h_return,nb_evals,nb_iters))

# How many random initialisation points are tested. 
n = 50

ParetoOpt = np.zeros((n,4))
ParetoOut = np.zeros((n,8))
elapsed = np.zeros((n,1))

# Start the optimization 
# This loop creates finds the first three points on the Pareto front
for i in range(0, n):
    x0 = np.random.rand(4) #
    t1 = time.time()
    params = ['BB_OUTPUT_TYPE OBJ PB PB','MAX_BB_EVAL 200','UPPER_BOUND * 1','LOWER_BOUND * 0']
    [ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = PyNomad.optimize(bb,x0,lb,ub,params)
    print ('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,f_return,h_return,nb_evals,nb_iters))
    elapsed = time.time() - t1
    ParetoOpt[i,:] = x_return
    ParetoOut[i,:] = np.transpose(np.concatenate((np.asarray(process(x_return, adsorbents[0])),elapsed,stopflag),axis=None))


# This loop fits the Pareto front into a GP regression model 
# The next opitization point is created at the maximum variance 
fig1, ax1 = plt.subplots()      
ax1.set_title('NOMAD')
ax1.boxplot(ParetoOpt,labels=['p_max','p_int','p_min','Tads'])
plt.show()

print(ParetoOpt)
print(ParetoOut)
with open('PyNOMAD_single.txt','w') as f:
    f.write(f'[{adsorbents}] \n')
    f.write(f'optimal variables = {ParetoOpt} \n')
    f.write(f'optimal outputs = {ParetoOut} \n')




