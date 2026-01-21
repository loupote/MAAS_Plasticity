import numpy as np
import matplotlib.pyplot as plt


class ReturnMapping:
    def __init__(self, Pale, initial_deltap, itermax, tol):
        self.pale = Pale
        self.initial_deltap = initial_deltap 
        self.max_iter = itermax
        self.tolerence = tol

    def dp_function(self, dp, p_n, sigma_pred):
        # Fonction scalaire non linéaire dont il faut trouver le zéro 'dp'
        # Eq (4.25)
        f = abs(sigma_pred) - self.pale.E*dp - self.pale.R(p_n+dp) - self.pale.sigma_y
        df = - self.pale.E - self.pale.B*self.pale.m*(p_n+dp)**(self.pale.m-1.0) # Dérivée par rapport à 'dp' évaluée
        return f, df

    def solve_dp_function(self, f, p_n, sigma_pred):
        # Algortihme de Newton-Raphson
        n = 0
        deltap = self.initial_deltap
        res, df = f(deltap, p_n, sigma_pred)
        while n < self.max_iter and abs(res) > self.tolerence:
            deltap -= res/df
            res, df = f(deltap, p_n, sigma_pred)
            n += 1
        return deltap


    def tangent_operator(self, p_next, sigma_next):
        # Opérateur tangent discret
        # Equation (4.35)
        dpdeps = self.pale.E/(self.pale.E + self.pale.diff_R(p_next))*np.sign(sigma_next)
        H = self.pale.E * (1 - dpdeps * np.sign(sigma_next))
        return H


    def retourRadial(self, sigma_n, delta_eps, ep_n, p_n):

        #Prédiction élastique
        sigma_pred = sigma_n + self.pale.E*delta_eps
        f = self.pale.critere(sigma_pred, self.pale.R(p_n))


        if f <= 0.0: # La prédiction élastique est admissible
            sigma_next = sigma_pred
            ep_next    = ep_n
            p_next     = p_n
            H_next     = self.pale.E


        else: # Correction plastique
            delta_p = self.solve_dp_function(self.dp_function, p_n, sigma_pred)

            p_next = p_n + delta_p
            sigma_next = sigma_pred - self.pale.E*delta_p*np.sign(sigma_pred)
            ep_next = ep_n + delta_p*np.sign(sigma_pred) # Eq. (4.26)
            H_next = self.tangent_operator(p_next, sigma_next)

        return sigma_next, ep_next, p_next, H_next
    
    def plot_sigma_eps(self, number_steps):
        eps_max = 0.02

        # Scénario de chargement du matériau
        stretching               = np.linspace(0.0, eps_max, number_steps)
        unstretching_compressing = np.linspace(eps_max, -eps_max, number_steps)
        decompressing            = np.linspace(-eps_max, 0.0, number_steps)
        eps_values               = np.concatenate((stretching, unstretching_compressing, decompressing))

        stress_values            = np.zeros(3*number_steps)
        eps_plat_values          = np.zeros(3*number_steps)
        p_values                 = np.zeros(3*number_steps)

        for i in range(1, 3*number_steps):
            delta_eps = eps_values[i] - eps_values[i-1]
            stress_values[i], eps_plat_values[i], p_values[i], _ = self.retourRadial(stress_values[i-1], delta_eps, eps_plat_values[i-1], p_values[i-1])
        it=0 

        fig, axes = plt.subplots(1, 3, figsize=(25, 5))
        fig.suptitle("Isotropic hardening")
        ax = axes[0]
        ax.plot(eps_values, stress_values) 
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (Pa)")
        ax.set_title(f"sigma-eps")
        ax.grid(True)


        ax = axes[1]
        ax.plot(eps_plat_values, stress_values) 
        ax.set_xlabel("Plastic strain")
        ax.set_ylabel("Stress (Pa)")
        ax.set_title(f"Plastic strain evolution")
        ax.grid(True)

        ax = axes[2]
        ax.plot(p_values, stress_values) 
        ax.set_xlabel("Cumulated plasticity")
        ax.set_ylabel("Stress (Pa)")
        ax.set_title(f"p evolution")
        ax.grid(True)
        plt.show()