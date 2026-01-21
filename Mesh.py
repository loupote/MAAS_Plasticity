import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, Nelem, L):
        self.Nelem  = Nelem
        self.Nnodes = Nelem + 1                   # Nombre de noeuds
        self.Le     = L/Nelem                     # Longueur d'un élément
        self.D      = np.linspace(0, L, self.Nnodes)   # Représentation du domaine 1D

        # Affichage du maillage 1D
        plt.figure()
        plt.plot(self.D, np.zeros(self.Nnodes), 'b-x')
        plt.title('Maillage')
        plt.xlabel('x(m)')
        plt.show()
    