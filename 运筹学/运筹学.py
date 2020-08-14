import numpy as np
import matplotlib.pyplot as plt

def plotparam(param_range, label, a1=200, a2=100, b1=20, b2=20, c=10,
              alpha=0.6, gamma=0.95, xi1=0.5, xi2=0.5, lambda1=0.6, lambda2=0.4, phi=0.6):
    for param in param_range:
    globals()[label] = param
    k11=2*(1-xi1+phi*xi1)*(b1+c*lambda1*lambda1)/(1-xi1)-\
        (1-alpha)*gamma*c*lambda1/(1-alpha*gamma)-c*lambda1*lambda1
    k12=-(2*phi*xi1*(b1+c*lambda1*lambda1)/(1-xi1)+2*c*lambda1*lambda1+b1)+\
        (1-alpha)*gamma*c*lambda1/(1-alpha*gamma)
    k13=c*lambda1*lambda2*(2+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))-\
        (1-alpha)*gamma*c*lambda2/(1-alpha*gamma)-c*lambda2*lambda1
    k14=-c*lambda1*lambda2*(2+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))+\
        (1-alpha)*gamma*c*lambda2/(1-alpha*gamma)
    
    k21=2*phi*xi1*(b1+c*lambda1*lambda1)/(1-xi1)+b1+c*lambda1*lambda1
    k22=-2*(1-xi1+phi*xi1)*(b1+c*lambda1*lambda1)/(1-xi1)
    k23=c*lambda1*lambda2*(1+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))
    k24=-c*lambda2*lambda1*(2+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))
    
    k31=c*lambda2*lambda1*(2+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))-\
        (1-alpha)*gamma*c*lambda1/(1-alpha*gamma)-c*lambda1*lambda2
    k32=-c*lambda2*lambda1*(2+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))+\
        (1-alpha)*gamma*c*lambda1/(1-alpha*gamma)
    k33=2*(1-xi2+phi*xi2)*(b2+c*lambda2*lambda2)/(1-xi2)-\
        (1-alpha)*gamma*c*lambda2/(1-alpha*gamma)-c*lambda2*lambda2;
    k34=-(2*phi*xi2*(b2+c*lambda2*lambda2)/(1-xi2)+2*c*lambda2*lambda2+b2)+\
        (1-alpha)*gamma*c*lambda2/(1-alpha*gamma)
    
    k41=c*lambda2*lambda1*(1+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))
    k42=-c*lambda1*lambda2*(2+phi*xi2/(1-xi2)+phi*xi1/(1-xi1))
    k43=2*phi*xi2*(b2+c*lambda2*lambda2)/(1-xi2)+b2+c*lambda2*lambda2
    k44=-2*(1-xi2+phi*xi2)*(b2+c*lambda2*lambda2)/(1-xi2)

    K=np.array([[k11,k12,k13,k14],[k21,k22,k23,k24],
                [k31,k32,k33,k34],[k41,k42,k43,k44]])
    p11, p12, p21, p22 = ([], [], [], [])
    A=np.array([[a1],[0],[a2],[0]])
    P=np.dot(np.linalg.inv(K),A)
    p11.append(P[0,0])
    p12.append(P[1,0])
    p21.append(P[2,0])
    p22.append(P[3,0])
    #pij指第i个产品第j个阶段
    fig, ax = plt.subplots(1)
    ax.plot(param_range,p11)
    ax.plot(param_range,p12)
    ax.plot(param_range,p21)
    ax.plot(param_range,p22)



plotparam(param_range=range(100, 210, 10), param)

