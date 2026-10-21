# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:24:59 2024

@author: NAAvd
"""

from math import sqrt
from math import sin
from math import cos
from math import fabs
from math import exp
from math import log
import numpy as np 
import math as math 
import matplotlib.pyplot as plt 
from multiprocess import Pool

import scipy
from scipy import special
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from sympy.solvers import solve

un =70000
u = int(un)

####################основные функции потенциала и умноженную на Г-Б
def potential(phi,V0,alp):
    """ Here we define the potential of the scalar field phi
        input value if phi and the return values are v and dvdphi.
        It is best to keep this definition in a separate module.
        Easier to edit if the potential is changed."""
    # alp = 0.5
    v=0.5*V0*abs(phi)**alp
    dvdphi=0.5*alp*V0*abs(phi)**(alp-1)*phi/abs(phi)
    # dvdphi2=0.5*alp*(alp-1)*V0*abs(phi)**(alp-2)*phi/abs(phi)
    return v,dvdphi

def GBfunc(phi,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    V_PHIB2 = 0.5*alp*(alp-1)*V0*abs(phi)**(alp-2)*phi/abs(phi)
    f=C/(lamb+VB)/4
    df = -C*V_PHIB/(VB+lamb)**2/4
    df2 = (2*C*V_PHIB**2-C*V_PHIB2*(VB+lamb))/(VB+lamb)**3/4
    return f,df,df2

#####################функция для фона

def background(y,t,V0,alp,C,lamb):
    VB,V_PHIB=potential(y[0],V0,alp)
    f,df,df2=GBfunc(y[0],V0,alp,C,lamb)
    H = y[2]
    dphi = y[1]

    
    Hd = H*(df2*dphi**2*H**2-4*df*dphi*H**2-df*V_PHIB-12*H**4*df**2-
            0.25*dphi**2)/(0.5-2*df*dphi*H**2+12*H**4*df**2)  
    
    d2phi = -3*dphi-V_PHIB/H**2-12*df*H**2-(dphi/H+12*df*H)*Hd
    
    return(dphi,d2phi,Hd)

####################слоурольный параметр

def epsilon_0(phi,dphi,H,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)
    Hd = H*(df2*dphi**2*H**2-4*df*dphi*H**2-df*V_PHIB-12*H**4*df**2-
              0.25*dphi**2)/(0.5-2*df*dphi*H**2+12*H**4*df**2)
    eps = -Hd/H
    return(eps)

#######################вычисление фона

efolds=np.arange(0,100,1e-4)

############ ввод констант теории ##############
V0 = 4.05e-11*2
alp = 2
C = 2.754
lamb = 1.0125e-12
################################################

y0 = np.zeros(3)
# y0[0] = 2.7565
y0[0] = 3.5
# y0[1] = -6.467472121356432e-07

VB,V_PHIB=potential(y0[0],V0,alp)
f,df,df2=GBfunc(y0[0],V0,alp,C,lamb)

H = sqrt(VB/3)
y0[1] = -(V_PHIB+12*df*H**4)/(3*H**2)

D = (6-y0[1]**2)**2 - 192*VB*df*y0[1]
y0[2] = sqrt((6-y0[1]**2-sqrt(D))/(48*df*y0[1]))


sol = odeint(background, y0, efolds, args=(V0,alp,C,lamb,),atol=1e-8)
phi=sol[:,0]
dphi=sol[:,1]
Hsol = sol[:,2]

eps0 = epsilon_0(phi, dphi, Hsol, V0, alp, C, lamb)

################построение графиков для фоновых функций

fig, ax = plt.subplots()

plt.plot(efolds, phi, 'red')

ax.set_xlabel("N")
ax.set_ylabel("phi")

plt.savefig('fig_01.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds, dphi, 'blue')

ax.set_xlabel("N")
ax.set_ylabel("dphi")

plt.savefig('fig_02.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds, Hsol, 'green')

ax.set_xlabel("N")
ax.set_ylabel("H")

plt.savefig('fig_03.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds, eps0, 'brown')

ax.set_xlabel("N")
ax.set_ylabel("eps0")

plt.savefig('fig_04.png', dpi=150); plt.close()

################### определение различных граничных условий ####################

############# определение момента времени после окончания инфляции с небольшим запасом ###########
stop = 1
i=0
while(stop!=0)&(i<len(efolds)):
    if (eps0[i]>1.0001)&(i>3000):
        stop = 0
    i+=1
    
###### переопределение Хаббла и ефолдов, чтобы избежать численных ошибок ######

efolds_1 = np.zeros(i)
Hsol_1 = np.zeros(len(efolds_1))
j=0
while (j<i):
    efolds_1[j] = efolds[j]
    Hsol_1[j] = Hsol[j]
    j+=1
print(len(efolds_1),'   ',len(efolds))    

############ определение момента окончания инфляции ############

stop = 1
i = 0
while (stop!=0)&(i<len(efolds_1)):
    if (eps0[i]>1)&(i>3000):
        stop = 0
    i+=1
iend = i-1

######### определение момента выхода возмущений за горизонт(50 ефолдов до конца) ##########

stop = 1
i = 0
iout = 0
while (stop!=0)&(i<len(efolds_1)):
    if efolds_1[i]/(efolds_1[iend]-65)>1:
        iout = i
        stop = 0
    i+=1
    
######### нормировка или нахождение начального значения масштабного фактора ############

kpivot = 0.05
ai=kpivot/(np.exp(efolds_1[iout])*Hsol[iout])

######### момент начала численного счёта возмущений ########### ??????????????????

stop = 1
i = 0
while (stop!=0)&(i<len(efolds_1)):
    if 100*ai*exp(efolds_1[i])*Hsol[i]/kpivot>1:
        i_in = i-1
        stop = 0
    i+=1
    
########## момент окончания численного счёта возмущений ########### ???????????????????

stop = 1
i = 0
while (stop!=0)&(i<len(efolds_1)):
    if 1e-5*ai*exp(efolds_1[i])*Hsol[i]/kpivot>1:
        i_en = i-1
        stop = 0
    i+=1
    
#################### функция для возмущений ################################

def с_S_func(dphi,phi,H,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)
    dtf = df*dphi*H
    Hd = H*(df2*dphi**2*H**2-4*df*dphi*H**2-df*V_PHIB-12*H**4*df**2-0.25*dphi**2)/(0.5-2*df*dphi*H**2+12*H**4*df**2)
    d2phi = -3*dphi-V_PHIB/H**2-12*df*H**2-(dphi/H+12*df*H)*Hd
    dtf2 = df2*dphi**2*H**2 +df*(d2phi*H**2+dphi*Hd*H)
    с_S = 1 + 4*dtf*16*H**4*dtf**2*(dtf2/dtf-H+Hd*H*(1-4*H*dtf)/(H**2*dtf))/(1-4*H*dtf)**2/(H**2*dphi**2 + 
                                1.5*16*H**4*dtf**2/(1-4*H*dtf))
    return(с_S)

def perturbation(all_var,t,k,ai,V0,alp,C,lamb):
    dphi, phi, H, dksi_re,  ksi_re, dksi_im, ksi_im, dh_re, h_re, dh_im, h_im = all_var
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)

    Hd = H*(df2*dphi**2*H**2-4*df*dphi*H**2-df*V_PHIB-12*H**4*df**2-0.25*dphi**2)/(0.5-2*df*dphi*H**2+12*H**4*df**2)
    d2phi = -3*dphi-V_PHIB/H**2-12*df*H**2-(dphi/H+12*df*H)*Hd
    
    dtf = df*dphi*H
    dtf2 = df2*dphi**2*H**2 +df*(d2phi*H**2+dphi*Hd*H)
                               
    G_S = (3 - 12*H*dtf+(-3*H**2+0.5*dphi**2*H**2+24*H**3*dtf)*((1-4*H*dtf)/(H*(1-6*H*dtf)))**2)

    dotG_S = (-12*Hd*H*dtf-12*H*dtf2+(-6*Hd*H**2+H*dphi*(H**2*d2phi+dphi*Hd*H)+
        72*H**3*Hd*dtf+24*H**3*dtf2)*((1-4*H*dtf)/(H*(1-6*H*dtf)))**2+
    (-3*H**2+0.5*H**2*dphi**2+24*H**3*dtf)*2*(1-4*H*dtf)/(H*(1-6*H*dtf))*(H*(1-
    6*H*dtf)*(-4*Hd*H*dtf-4*H*dtf2) - (1-4*H*dtf)*(Hd*H-12*H**2*Hd*dtf-
    6*H**2*dtf2))/(H**2*(1-6*H*dtf)**2))
                           
    
    F_S =( (1-4*H*dtf)**2/(1-6*H*dtf) + 2*(1-4*H*dtf)*(-4*Hd*H*dtf-4*H*dtf2)/(H*(1-
         6*H*dtf)) - (1-4*H*dtf)**2*(Hd*H-12*Hd*H**2*dtf-6*H**2*dtf2)/(H*(1-6*H*dtf))**2
          -1+4*dtf2)
    
    c_S2 = с_S_func(dphi,phi,H,V0,alp,C,lamb)
    a = ai*exp(t)

    d2ksi_re = -(3+dotG_S/G_S/H+Hd/H)*dksi_re-k**2*c_S2/(a**2*H**2)*ksi_re
    d2ksi_im = -(3+dotG_S/G_S/H+Hd/H)*dksi_im-k**2*c_S2/(a**2*H**2)*ksi_im
    
    G_T = (1-4*dtf*H)
    F_T = (1-4*dtf2)
    c2T = F_T/G_T
    dotG_T = -4*dtf2*H-4*dtf*H*Hd
    
    d2h_re = -(3+dotG_T/G_T/H+Hd/H)*dh_re-k**2*c2T/(a**2*H**2)*h_re
    d2h_im = -(3+dotG_T/G_T/H+Hd/H)*dh_im-k**2*c2T/(a**2*H**2)*h_im
    return[d2phi, dphi, Hd, d2ksi_re, dksi_re, d2ksi_im, dksi_im, d2h_re, dh_re, d2h_im, dh_im]    

####################### поиск ближайшего значения к заданному ###############################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def boundary(k,t,aH,Nic_cond,Nshs_cond):
    ic_diff=k-Nic_cond*aH
    shs_diff=k-Nshs_cond*aH
    idx_ic,ah_ic=find_nearest(ic_diff,0)
    idx_shs,ah_shs=find_nearest(shs_diff,0)
    Nic=t[idx_ic]
    Nshs=t[idx_shs]
    return idx_ic,Nic,Nshs

############# 

def G_S_func(dphi,phi,H,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)
    dtf = df*dphi*H
    G_S = (3 - 12*H*dtf+(-3*H**2+0.5*dphi**2*H**2+24*H**3*dtf)*((1-4*H*dtf)/(H*(1-6*H*dtf)))**2)
    return(G_S)

def dG_S_func(dphi,phi,H,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)
    dtf = df*dphi*H
    Hd = H*(df2*dphi**2*H**2-4*df*dphi*H**2-df*V_PHIB-12*H**4*df**2-
            0.25*dphi**2)/(0.5-2*df*dphi*H**2+12*H**4*df**2)
    d2phi = -3*dphi-V_PHIB/H**2-12*df*H**2-(dphi/H+12*df*H)*Hd
    dtf2 = df2*dphi**2*H**2 +df*(d2phi*H**2+dphi*Hd*H)
    dotG_S = (-12*Hd*H*dtf-12*H*dtf2+(-6*Hd*H**2+H*dphi*(H**2*d2phi+dphi*Hd*H)+
        72*H**3*Hd*dtf+24*H**3*dtf2)*((1-4*H*dtf)/(H*(1-6*H*dtf)))**2+
    (-3*H**2+0.5*H**2*dphi**2+24*H**3*dtf)*2*(1-4*H*dtf)/(H*(1-6*H*dtf))*(H*(1-
    6*H*dtf)*(-4*Hd*H*dtf-4*H*dtf2) - (1-4*H*dtf)*(Hd*H-12*H**2*Hd*dtf-
    6*H**2*dtf2))/(H**2*(1-6*H*dtf)**2))
    return(dotG_S)

def G_T_func(dphi,phi,H,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)
    dtf = df*dphi*H
    G_T = (1-4*dtf*H)
    return(G_T)

def dG_T_func(dphi,phi,H,V0,alp,C,lamb):
    VB,V_PHIB=potential(phi,V0,alp)
    f,df,df2=GBfunc(phi,V0,alp,C,lamb)
    dtf = df*dphi*H
    Hd = H*(df2*dphi**2*H**2-4*df*dphi*H**2-df*V_PHIB-12*H**4*df**2-
            0.25*dphi**2)/(0.5-2*df*dphi*H**2+12*H**4*df**2)
    d2phi = -3*dphi-V_PHIB/H**2-12*df*H**2-(dphi/H+12*df*H)*Hd
    dtf2 = df2*dphi**2*H**2 +df*(d2phi*H**2+dphi*Hd*H)
    dotG_T = -4*dtf2*H-4*dtf*H*Hd
    return(dotG_T)

def z(a, G_S):
    return(a*(2*G_S)**0.5)

def zH(a, G_T):
    return(a*(G_T)**0.5)

############
i = 0

P_s = np.zeros(1000)
P_t = np.zeros(1000)
logP_s = np.zeros(1000)
logP_t = np.zeros(1000)
r = np.zeros(1000)
logk = np.zeros(1000)
kkk = np.zeros(1000)
rr = 0

for i in range(1000):
    k = 0.01 + 0.0001*i
    i_in,Nic,Nshs=boundary(k,efolds_1,ai*np.exp(efolds_1)*Hsol_1,100,1e-5)
    G_S1 = G_S_func(dphi[i_in],phi[i_in],Hsol_1[i_in], V0,alp,C,lamb)
    G_T1 = G_T_func(dphi[i_in],phi[i_in],Hsol_1[i_in], V0,alp,C,lamb)
    dG_S1 = dG_S_func(dphi[i_in],phi[i_in],Hsol_1[i_in], V0,alp,C,lamb)
    dG_T1 = dG_T_func(dphi[i_in],phi[i_in],Hsol_1[i_in],V0,alp,C,lamb)
    
    ksi_re = (1/(np.sqrt(2*k)))/z(ai*exp(efolds_1[i_in]),G_S1)
    h_re = (1/(np.sqrt(2*k)))/zH(ai*exp(efolds_1[i_in]),G_T1)
    
    ksi_im = 0
    h_im = 0
    
    dksi_re = -ksi_re*(dG_S1/(2*Hsol_1[i_in]*G_S1)+1)
    dh_re = -h_re*(dG_T1/(2*Hsol_1[i_in]*G_T1)+1)
    
    dksi_im = -ksi_re*k/(ai*exp(efolds_1[i_in])*Hsol_1[i_in])
    dh_im = -h_re*k/(ai*exp(efolds_1[i_in])*Hsol_1[i_in])
    
    all_var = [dphi[i_in], phi[i_in], Hsol_1[i_in], dksi_re, ksi_re, dksi_im, ksi_im, dh_re, h_re, dh_im, h_im]
    efolds_pert=np.arange(Nic,Nshs, 0.0001)
    
    sol_1 = odeint(perturbation, all_var, efolds_pert, args=(k,ai,V0,alp,C,lamb,),atol=1e-8)
    
    ksi_r = sol_1[-1,4]
    ksi_i = sol_1[-1,6]
    h_r = sol_1[-1,8]
    h_i = sol_1[-1,10]
    
    ksi_re = sol_1[:,4]
    ksi_im = sol_1[:,6]
    h_re = sol_1[:,8]
    h_im = sol_1[:,10]
    
    P_s[i] = (k**3/(2*np.pi**2)*((ksi_r*ksi_r)+(ksi_i*ksi_i)))
    P_t[i] = (8*k**3/(2*np.pi**2)*((h_r*h_r)+(h_i*h_i)))
    # print(i)
    r[i] = P_t[i]/P_s[i]
    logP_s[i] = log(P_s[i])
    logP_t[i] = log(P_t[i])
    logk[i] = log(k)
    kkk[i] = k
    rr+= r[i]
    
A = np.vstack([logk, np.ones(len(logk))]).T
n_s, c = np.linalg.lstsq(A, logP_s)[0]


n_t, c1 = np.linalg.lstsq(A, logP_t)[0]
print('r = ', r[400], 'n_s = ', n_s+1, 'n_t=',n_t, 'P_s = ', P_s[i-1], 'P_t = ', P_t[i-1])

##############

fig, ax = plt.subplots()

plt.plot(kkk, r, 'brown')

ax.set_xlabel("k")
ax.set_ylabel("r")

plt.savefig('fig_05.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(logk, logP_s, 'brown')

ax.set_xlabel("logk")
ax.set_ylabel("logP_s")

plt.savefig('fig_06.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(logk, logP_t, 'brown')

ax.set_xlabel("logk")
ax.set_ylabel("logP_t")

plt.savefig('fig_07.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds_pert, ksi_re, 'brown')

ax.set_xlabel("N")
ax.set_ylabel("ksi_r")

plt.savefig('fig_08.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds_pert, ksi_im, 'brown')

ax.set_xlabel("N")
ax.set_ylabel("ksi_i")

plt.savefig('fig_09.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds_pert, h_re, 'brown')

ax.set_xlabel("N")
ax.set_ylabel("h_r")

plt.savefig('fig_10.png', dpi=150); plt.close()

fig, ax = plt.subplots()

plt.plot(efolds_pert, h_im, 'brown')

ax.set_xlabel("N")
ax.set_ylabel("h_i")

plt.savefig('fig_11.png', dpi=150); plt.close()
