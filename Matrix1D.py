'''
Author: Thomas HEUZE
Centrale Nantes, France
MAAS Elastoplasticity course, 2022
'''

import numpy as np

def Lin_shp(idx, ksi, Xe):
    # Linear shape function on 1D parent element
    # ksi can be an array of length n, so the function will return an array (n, 2)
    if (idx == 1):
        # N0, N1
        return np.array([(1.0-ksi)/2.0, (1.0+ksi)/2.0])
    elif (idx == 2):
        # dN0/dx, dN1/dx
        return 1/Jac(Xe)*(np.array([-np.ones(len(ksi)), np.ones(len(ksi))])/2.0)

def Lin_eval(ksi, node_value):
    # Interpolation evaluation of a field quantity
    return (node_value[0]*(1.0-ksi) + node_value[1]*(1.0+ksi))/2.0

def Cst_eval(ksi, node_value):
    return node_value*np.ones(len(ksi))

def Jac(Xe):
    # 1D jacobian value
    # dx/dksi = dN0/dksi*x0 + dN1/dksi*x1 = -1/2*x0 + 1/2*x1
    return ((Xe[1]-Xe[0])/2.0)

def Matrix1D(x, op1, op2, lin_fields, cst_fields):
    Nnodes = len(x)
    Nelem = Nnodes - 1

    # Connectivity array
    T10 = np.array([np.arange(0,Nnodes-1,1), np.arange(1,Nnodes,1)]).T
    
    # Gauss points to get the exact result for polynomial functions under 5th order:
    # -sqrt(0.6); 0; sqrt(0.6)
    ksi_integ = np.array([0.774596669241483, 0.0, -0.774596669241483])
    # 5/9; 8/9; 5/9
    w_integ = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    A = np.zeros((Nnodes, Nnodes))
    for i in range(Nelem):
        mapp = T10[i,:] # To get the the nodes linked to the elements (mapping)
        Xe = x[mapp] # To get the coordinates of the nodes from the considered element
        loc_jac = Jac(Xe) # Local jacobian 
        
        M1 = Lin_shp(op1+1, ksi_integ, Xe)
        M2 = Lin_shp(op2+1, ksi_integ, Xe)
        D = np.array(w_integ)
        if len(lin_fields) != 0:
            D *= Lin_eval(ksi_integ, lin_fields[mapp])
        if len(cst_fields) != 0:
            D *= Cst_eval(ksi_integ, cst_fields[i])

        # Assembling
        A[mapp[0]:mapp[1]+1, mapp[0]:mapp[1]+1] += (loc_jac*np.dot(M1, np.dot(np.diag(D), M2.T)))
    return A
