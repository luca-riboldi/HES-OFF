"""
This script optimizes the PSA function in terms of minimizing the avoided cost with constraints on purity and recovery.
The optimization can be done in two different ways: by using opt_one_guess and by using opt_multi_guess. 

opt_one_guess performes a single optimization using scipy.minimize with the method SLSQP with a large tolerance. 
The starting guess is the same each time: at the midpoint between the bounds for all inputs. This means that this
optimizer will be relativly quick, but somewhat crude. It is meant as a way of making a first sorting of materials, so that one can 
quickly throw away materials that do not meet the constraints and materials that is way to expensive. It will, however, not always 
find the global minimum of the function, so one should be carefull not to draw conclusions between materials that seems to perform
somewhat similar.

opt_multi_guess uses the same method as opt_one_guess, but with a much smaller tolerance. It does the optimization several times with different
starting guesses, set as random numbers between 0 and 1. The optimization stops when 4 minimizations in a row does not result in a
better result, or when 20 minimizations is ran. 
"""

from psa_basic_mc_fast_4cycle_TEA_mod import PSA
from scipy.optimize import minimize
from scipy.optimize import differential_evolution 
from scipy.optimize import basinhopping, shgo, brute
import numpy as np
import time
from scipy.optimize import LinearConstraint
import scipy.optimize as optimize
from scipy.optimize import Bounds
from multiprocessing import Pool
import concurrent.futures

#-----------------------------------------------------------
# Part I: Specify all parameters required by the script.
#-----------------------------------------------------------

steps = 400 #2000 #

# Define the PSA scenario that we wish to optimize: 
case_study = 'coal' #'cement' #'hydrogen' #'natural_gas_onshore' #natural_gas_offshore
if case_study == 'coal':
    #from coal case study
    y_feed = 0.1521
    p_feed = 1.016
    T_feed = 50+273.15 
elif case_study == 'cement':
    #from cement case study
    y_feed = 0.247
    p_feed = 1.01
    T_feed = 130 + 273.15
elif case_study == 'hydrogen':
    # #from hydrogen case study
    y_feed = 0.20
    p_feed = 1.02
    T_feed = 353.15 #298.15 #
elif case_study == 'natural_gas_onshore':
    #from natural gas case study
    y_feed = 0.0432  
    p_feed = 1.01
    T_feed = 86.8+273.15    
elif case_study == 'natural_gas_offshore':
    #from natural gas case study
    y_feed = 0.0407  
    p_feed = 1.01
    T_feed = 113+273.15 
elif case_study == 'waste':
    #from waste case study
    y_feed = 0.1361 
    p_feed = 1.01
    T_feed = 160+273.15 

# Define process costraints.
purity = 0.95 #
recovery = 0.80 #

#Define bounds
p_H_min = 1.0
p_H_max = 1.4 #
p_I_min = 0.21 #0.2 + 0.01
p_I_max = 0.7 #0.4 #
p_L_min = 0.02 
p_L_max = 0.20
T_ads_min = 298
T_ads_max = 320#360

bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

#-----------------------------------------------------------
# Part II: Specify the cost objectives and the two constraints constraints.
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

def constraint_purity(x, adsorbent):
    """Constraint: Minimum purity requirements."""
    return 100*(process(x, adsorbent)[0]-purity)

def constraint_recovery(x, adsorbent):
    """Constraint: Minimum recovery requirements."""
    return 100*(process(x, adsorbent)[1]-recovery)

def objective_avoided_cost(x, adsorbent):
    return process(x, adsorbent)[4] / 100

def objective_working_capacity(x):
    """Objective: Maximize the working capacity."""
    return 1/process(x, adsorbent)[2]

def objective_work(x):
    """Objective: Miminize the specific work."""
    return process(x, adsorbent)[3]/10000

def objective_mass_ads(x):
    """Objective: Miminize the specific mass of adsorbent."""
    return process(x, adsorbent)[5]

#-----------------------------------------------------------
# Part III: Define optimizer functions
#-----------------------------------------------------------

def opt_one_guess(a):
    """
    Performs a simple run of the gradient based method "SLSQP" with initial guess that is the midle of the bounds for all materials.
    Both the step length "eps" and the function tolerance "ftol" is set reativly high to get a fast and somewhat crude optimization
    Input:
    a : name of adsorbent
    Returns:
    pr[4]: the cost found by the optimizer to be the minimal cost, only if optimization succeeded. Returns 0 else
    """
    #make the multi-component data for the given adsorbent. Needs only to be ran the first time for each material.
    #if this is not the first time for the adsorbent, the function call should be commented out to save time
    #make_multi_comp(a)

    #the constraints must be defined for each adsorbent 
    cons = ({'type': 'ineq', 'fun': constraint_purity, 'args' : (a,)},
        {'type': 'ineq', 'fun': constraint_recovery, 'args' : (a,)})

    #use initial guesses in the middle of the bounds    
    init = np.random.rand(4) #[0.5] * 4 #

    result = minimize(objective_avoided_cost, init,args = (a,),  bounds = bounds, constraints = cons, method = "SLSQP"\
        , options = {'eps': 1e-2, 'maxiter': 25, 'ftol' : 1e-05, 'disp' : True}) #{'eps': 1e-2, 'maxiter': 25, 'ftol' : 1e-04, 'disp' : True}
    # result = minimize(objective_avoided_cost, init,args = (a,),  bounds = bounds, constraints = cons, method = "SLSQP"\
    #     , options = {'eps': 5e-2, 'maxiter': 15, 'ftol' : 1e-02, 'disp' : True}) #'eps': 1e-2,
    pr = process(result.x, a)
    print(f"{a}:\nInput:  {unnormalize( result.x )}\nOutput: Purity  {pr[0]:.2f}\nRecovery : {pr[1]:.2f} \
         \nSp.work :{pr[2]:.1f}\nAvoided cost: {pr[4]:.1f} ")

    print(f"{a}:\n optimization success?:  {result.success:.1f}")
    # print("success? :", result.success)
    if result.success:
        return pr[4]
    else:
        return 0

def opt_multi_guess(a):
    """
    Performs several runs of the gradient based method "SLSQP" with initial guess that are random.
    Both the step length "eps" and the function tolerance "ftol" is set low to to get an accurate representation of the global minimum.
    Input:
    a : name of adsorbent
    Returns:
    pr[4]: the cost found by the optimizer to be the minimal cost, only if optimization succeeded. Returns 0 else.

    NB - should be studied a bit more when new materials are available. Check how fast / how often it finds global minimum.

    """    

    #make the multi-component data for the given adsorbent. Needs only to be ran the first time for each material.
    #if this is not the first time for the adsorbent, the function call should be commented out to save time
    #make_multi_comp(a)

    #the constraints must be defined for each adsorbent 
    cons = ({'type': 'ineq', 'fun': constraint_purity, 'args' : (a,)},
        {'type': 'ineq', 'fun': constraint_recovery, 'args' : (a,)})

    best_cost = 1000
    count = 0
    for i in range(10):
        count += 1

        #use random intitial values
        init = np.random.rand(4)

        #minimize
        result = minimize(objective_avoided_cost, init, args = (a,), bounds = bounds,constraints = cons, method = "SLSQP"\
            , options = {'eps': 5e-2, 'maxiter': 25, 'ftol' : 1e-03, 'disp' : True}) #options = {'eps': 1e-3, 'maxiter': 30, 'ftol' : 1e-04, 'disp' : False}
        
        pr = process(result.x, a)

        #save the first result so that best_result is defined even if the optimizer always fails
        if i ==0 :
            best_result = result
        #save if improvement is larger than 0.5 (just to avoid extremely small improvements)
        if result.success and pr[4] < best_cost - 0.5:
            best_cost = pr[4]
            best_result= result
            count = 0
        
        #break loop if 3 iteratios passes without getting a better result
        if count  >= 4:
            break


    pr = process(best_result.x, a)        
    print(f"{a}:\n Input:  {unnormalize( best_result.x )}\n Output: Purity  {pr[0]:.2f}\nRecovery : {pr[1]:.2f} \
         \nSp.work :{pr[2]:.1f}\nAvoided cost: {pr[4]:.1f} ")

    if best_result.success:
        return pr[4]
    else:
        return 0


#list of adsorbents. can be as long we want. 
adsorbents = ['Zeolite_40C'] ##'ALPMOF_40C' #'Zeolite_40C' #'UIO-66-NH2_40C' #'ZIF-8_40C' #'Zeolite_UOA' #'IISERP_UOA' #'UTSA16_UOA' 
#adsorbent = "ALPMOF_40C"

#time the optimization
t1 = time.time()
#--------------------------------------------------------------------
# Part IV: Optimize using the desired optimizing function in parallel
#--------------------------------------------------------------------

#run multiprocessing
if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor: 
        #divide the adsorbents in the list "adsorbents" on different  cores.
        # results will be in same order as the adsorbents 
        results = executor.map(opt_one_guess, adsorbents) #results = executor.map(opt_multi_guess, adsorbents) #
        # results = [opt_one_guess(ads) for ads in adsorbents] #Not parallelized

        #iterate through results and print minimization result. For large database: should probably be saved to a another file instead
        for i,r in enumerate(results):
            print(f"RESULT FOR ADSORBENT {adsorbents[i]} : {r}")

        
    print(f"Time used : { time.time() - t1 } seconds")

