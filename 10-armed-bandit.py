import numpy as np
import matplotlib.pyplot as plt

q_star = np.random.normal(loc = 0.0, scale = 1.0, size = 10)
# q_star(a), a = [1, 10], a normal Gaussian distribution with mean 0 and variance 1



def k_armed_bandit_one_run(qstar,epsilon,nStep):
    """
    One run of K-armed bandit simulation.
    Here, K is not explicitly specified, instead it is derived from the length qstar
    """
    
    K     = len(qstar)
    Q     = np.zeros(K)                   # Estimation of action value.
    actCnt  = np.zeros(K,dtype='int')     # Record the number of action#k being selected
 
    a     = np.zeros(nStep+1,dtype='int') # Record the adopted action in each time step.
    r     = np.zeros(nStep+1)             # Recode the reward in each time step
    
    optAct   = np.argmax(qstar)           # The ground-truth optimal action, with the largest qstar value
    optCnt   = 0                          # Count the number of time steps in which the optimal action is selected
    optRatio = np.zeros(nStep+1,dtype='float') # Item#0 for initialization
 
    for t in range(1,nStep+1): # loop over time step
        #1. action selection
        tmp = np.random.uniform(0,1)
        #print(tmp)
        if tmp < epsilon: # random selection for exploring 
            a[t] = np.random.choice(np.arange(K))
            #print('random selection: a[{0}] = {1}'.format(t,a[t]))
        else:             # greedy selection for exploitation
            #选择Q值最大的那个，当多个Q值并列第一时，从中任选一个--但是如何判断有多个并列第一的呢？
            #对Q进行random permutation处理后再找最大值可以等价地解决这个问题
            #因为np.argmax()是找第一个最大的(当存在多个同为最大时)    
            p = np.random.permutation(K)
            a[t] = p[np.argmax(Q[p])]
            #print('greedy selection: a[{0}] = {1}'.format(t,a[t]))
 
        actCnt[a[t]] = actCnt[a[t]] + 1
 
        #2. reward: draw from the normal distribution with mean = qstar[a[t]], and variance = 1.
        r[t] = np.random.randn() + qstar[a[t]]        
 
        #3.Update Q of the selected action - should be refined
        Q[a[t]] = (Q[a[t]]*(actCnt[a[t]]-1) + r[t])/actCnt[a[t]]    
        
        #4. Optimal Action Ratio tracking
        #print(a[t], optAct)
        if a[t] == optAct:
            optCnt = optCnt + 1
        optRatio[t] = optCnt/t
        
    return a,actCnt,r,Q,optRatio