import numpy as np
import Vector1D as vec
import Matrix1D as mat
import matplotlib.pyplot as plt


class PaleSolver():
    def __init__(self, op, M, itermax, tol, Pale, Mesh, ReturnMapping):
        self.max_iter     = itermax
        self.tolerence    = tol
        self.pale         = Pale
        self.mesh         = Mesh
        self.RR           = ReturnMapping

        # 0 for loading only, 1 for loading/unloading
        if op == 0:
            self.loading  = np.linspace(0.0, self.pale.omega_max, M)
            self.loading_steps = len(self.loading)
        elif op == 1:
            self.loading  = np.append(np.linspace(0.0, self.pale.omega_max, M),
                                           np.linspace(self.pale.omega_max, 0.0, M))
            self.loading_steps = len(self.loading)
        else:
            raise Exception("Option number is not correct. Please choose between 0 (loading only) and 1 (loading then unloading)")


        self.u            = np.zeros((self.loading_steps, self.mesh.Nnodes)) # Déplacements / Displacements
        self.sigma        = np.zeros((self.loading_steps, self.mesh.Nelem))  # Contraintes / Stress
        self.eps          = np.zeros((self.loading_steps, self.mesh.Nelem))  # Déformations / Infinitesimal strain
        self.fint         = np.zeros((self.loading_steps, self.mesh.Nnodes)) # Forces internes / Internal forces
        self.fext         = np.zeros((self.loading_steps, self.mesh.Nnodes)) # Forces externes / External forces
        self.eps_plas     = np.zeros((self.loading_steps, self.mesh.Nelem))  # Déformations plastiques / Plastic strains
        self.p            = np.zeros((self.loading_steps, self.mesh.Nelem))  # Déformation plastique cumulée
        self.res          = np.zeros((self.loading_steps, self.mesh.Nnodes)) # Résidu / Residual
        self.res_norm     = np.zeros(self.loading_steps)                     # Norme du résidu pour chaque chargement / Residual norm
        self.H            = np.zeros((self.loading_steps, self.mesh.Nelem))  # Opérateur tangent discret / Discrete tangent operator

        

    def ext_forces(self, dist, omega):
        # Forces volumiques (N/m)
        return self.pale.rho*self.pale.S*dist*omega**2

    def loadingLoop(self):
        # Boucle de chargement
        for t in range(1, self.loading_steps):
            # Candidat élastique (sigma = E*eps en 1D)
            Kelas = self.pale.E*self.pale.S*mat.Matrix1D(self.mesh.D, 1, 1, [], [])

            # Forces extérieures f_ext = integral de {N0(x), N1(x)}^T * f_L(x) dx
            self.fext[t, :] = vec.Linear_form(self.mesh.D, 0, self.ext_forces(self.mesh.D, self.loading[t]), [])
            
            # Calcul de u et eps
            #u0              = np.linalg.solve(Kelas[1:, 1:], self.fext[t, 1:] - Kelas[0, 0]*self.u[t, 0])
            #u0_test         = np.append(0, u0)
            delta_u         = np.linalg.solve(Kelas[1:, 1:], self.fext[t, 1:] - self.fint[t-1, 1:])
            u0              = self.u[t-1, 1:] + delta_u
            u_candidat      = np.append(0, u0)
            self.eps[t, :]  = (u_candidat[1:] - u_candidat[:-1])/self.mesh.Le
            delta_eps       = self.eps[t, :] - self.eps[t-1, :]
            
        
            # Intégration de la loi de comportement sur les éléments
            for elem in range(self.mesh.Nelem):
                # Puisque les fonctions sont linéaires, les déformations sont constantes sur l'élément
                # Une valeur sur chaque élément suffit alors pour représenter les différentes quantités
                self.sigma[t, elem], self.eps_plas[t, elem], self.p[t, elem], self.H[t, elem] = self.RR.retourRadial(self.sigma[t-1, elem], delta_eps[elem], self.eps_plas[t-1, elem], self.p[t-1, elem])
                
        
            # Calcul des forces intérieures une fois sigma admissible trouvé
            self.fint[t, :] = self.pale.S * vec.Linear_form(self.mesh.D, 1, [], self.sigma[t, :])
            

            # Calcul du résidu et de sa norme
            self.res[t, 1:]  = self.fext[t, 1:] - self.fint[t, 1:]
            self.res_norm[t] = np.linalg.norm(self.res[t, :])
            
            it = 0
            while self.res_norm[t] > self.tolerence and it < self.max_iter:
                # Domaine plastique
                
                # Calcul de la matrice tangente K = integral de B^T*H*B dV
                K = mat.Matrix1D(self.mesh.D, 1, 1, [], self.H[t, :]) * self.pale.S
                

                # Utilisation d'une matrice tangente élastique pour comparer les vitesses de convergence de Newton-Raphson
                '''
                K=Kelas
                '''
                
                delta_u        = np.linalg.solve(K[1:, 1:], self.res[t, 1:])
                u0            += delta_u
                u_candidat     = np.append(0, u0)
                self.eps[t, :] = (u_candidat[1:] - u_candidat[:-1])/self.mesh.Le
                delta_eps      = self.eps[t, :] - self.eps[t-1, :]
               
                # Intégration de la loi de comportement sur les éléments par Retour Radial (Return Mapping)
                for elem in range(self.mesh.Nelem):
                    self.sigma[t, elem], self.eps_plas[t, elem], self.p[t, elem], self.H[t, elem] = self.RR.retourRadial(self.sigma[t-1, elem], delta_eps[elem], self.eps_plas[t-1, elem], self.p[t-1, elem])
                    
                
                # Calcul des forces interieures
                self.fint[t, :]  = self.pale.S * vec.Linear_form(self.mesh.D, 1, [], self.sigma[t, :])

                # Calcul du residu
                self.res[t, 1:]  = self.fext[t, 1:] - self.fint[t, 1:]
                self.res_norm[t] = np.linalg.norm(self.res[t, :])
                
                it += 1

            print('Pas de chargement: ', t, ' | Itération: ', it, ' | Résidu', self.res_norm[t])
               
            self.u[t, :] = u_candidat # u_candidat est la solution trouvée dans le chargement en cours


    # Solutions analytiques élastiques
    def SigmaAnalytique(self, r, omega):
        sigma = self.pale.rho * omega**2/2 * (self.pale.L**2-r**2)
        return sigma

    def UAnalytique(self, r, omega):
        u = self.pale.rho * r * omega**2 / (2*self.pale.E) * (self.pale.L**2 - r**2/3)
        return u


    def plot_results(self, loading_idx):
        if loading_idx >= self.loading_steps:
            raise Exception(f"Loading index to plot should be less or equal to {self.loading_steps-1} (it is {loading_idx}).")
        x = np.linspace(0, self.pale.L, 150)

        fig, axes = plt.subplots(1, 3, figsize=(30, 5))
        ax = axes[0]
        ax.plot(x, self.SigmaAnalytique(x, self.loading[loading_idx]), label='Solution Analytique')
        ax.step(self.mesh.D[:-1], self.sigma[loading_idx, :], where='post', c='r', label='Solution EF')
        ax.legend()
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('Sigma (Pa)')
        ax.set_title(f'Contraintes le long de la pâle \n(chargement n°{loading_idx})')
        ax.grid(True)


        ax = axes[1]
        ax.plot(x, self.UAnalytique(x, self.loading[loading_idx]), label='Solution Analytique') 
        ax.plot(self.mesh.D, self.u[loading_idx, :], '-x', c='r', label='Solution EF')
        ax.legend()
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('U (m)')
        ax.set_title(f'Déplacements le long de la pâle \n(chargement n°{loading_idx})')
        ax.grid(True)

        ax = axes[2]
        ax.plot(self.loading, self.u[:, self.mesh.Nnodes-1], 'b-x')
        ax.set_xlabel('Vitesse de rotation (rad/s)')
        ax.set_ylabel('U (m)')
        plt.title('Déplacement en bout de pâle')
        ax.grid(True)
        
        plt.show()

        plt.plot(self.u[:, -1], self.fext[:, -1], 'x', linestyle='--', label='fext')
        plt.plot(self.u[:, -1], self.fint[:, -1], c='r', label='fint')
        plt.xlabel('U (m)')
        plt.ylabel('Forces (N)')
        plt.legend()
        plt.grid(True)
        plt.show()