import numpy as np
import math

class Pale:
    def __init__(self, young_modulus, cross_section, length, density, max_rotation, hardening_param1, hardening_param2, yield_):
        self.E           = young_modulus
        self.S           = cross_section
        self.L           = length
        self.rho         = density
        self.omega_max   = max_rotation*math.pi/30
        self.B           = hardening_param1
        self.m           = hardening_param2
        self.sigma_y     = yield_


    def R(self, p):
        # Loi d'évolution de l'écrouissage isotrope de type puissance
        return self.B*p**self.m

    def diff_R(self, p):
        # Loi d'évolution de l'écrouissage isotrope de type puissance
        return self.m*self.B*p**(self.m-1.0)

    def critere(self, sigma_pred, R):
        # Critère de Von Mises 1D comptant la loi d'écrouissage isotrope
        return abs(sigma_pred) - R - self.sigma_y