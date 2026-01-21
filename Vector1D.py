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
        # N0(ksi), N1(ksi)
        return np.array([(1.0-ksi)/2.0, (1.0+ksi)/2.0])
    elif (idx == 2):
        # dN0/dx, dN1/dx  = J^-1 * dN0/dksi, dN1/dksi
        return 1/Jac(Xe)*(np.array([-np.ones(len(ksi)), np.ones(len(ksi))])/2.0)

def Lin_eval(ksi, node_value):
    # Interpolation evaluation of a field quantity
    # field(ksi) = field(0)*N0(ksi) + field(1)*N1(ksi)

    # ksi (1D-array of any length): integration points
    # node_value (1D-array of length 2): node values
    # Returns: 1D-array of the same length as ksi

    return (node_value[0]*(1.0-ksi) + node_value[1]*(1.0+ksi))/2.0

def Cst_eval(ksi, node_value):
    return node_value*np.ones(len(ksi))

def Jac(Xe):
    # Local jacobian, depending on each real element
    # x(ksi) = N0(ksi)*x0 + N1(ksi)*x1
    # dx/dksi = dN0/dksi*x0 + dN1/dksi*x1 = -1/2*x0 + 1/2*x1
    return ((Xe[1]-Xe[0])/2.0)


def Linear_form(x, op, lin_fields, cst_fields):
    # Example: calculate the discrete body force value at the nodes
    # f_ext = S * integrale de N(x) * f_B(x) dx
    Nnodes = len(x)
    Nelem = Nnodes - 1

    # Connectivity array
    # Element 0 associated to the nodes 0 and 1, element 1 to the nodes 1 and 2 ...
    T10 = np.array([np.arange(0, Nnodes-1, 1), np.arange(1, Nnodes, 1)]).T
    
    # Gauss points to get the exact result for polynomial functions under 5th order:
    # -sqrt(0.6); 0; sqrt(0.6)
    ksi_integ = np.array([0.774596669241483, 0.0, -0.774596669241483])
    # 5/9; 8/9; 5/9
    w_integ = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    A = np.zeros(Nnodes)
    for elem in range(Nelem):
        mapp = T10[elem,:] # The two nodes number of the element
        
        D = np.array(w_integ)
        if len(lin_fields) != 0:
            D *= Lin_eval(ksi_integ, lin_fields[mapp]) # Value of the field at the 3 integration points (3,)
        if len(cst_fields) != 0:
            D *= Cst_eval(ksi_integ, cst_fields[elem]) # (3,)
        
        Xe = x[mapp]
        loc_jac = Jac(Xe)
        N = Lin_shp(op+1, ksi_integ, Xe) # Value of the shape functions N0 and N1, at the 3 integration points (3, 2)
        A[mapp[0]:mapp[1]+1] += np.dot(N, D.T) * loc_jac
    return A
