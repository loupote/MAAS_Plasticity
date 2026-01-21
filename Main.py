from Mesh import Mesh
from Pale import Pale
from PaleSolver import PaleSolver
from ReturnMapping import ReturnMapping

if __name__ == '__main__':

    # Paramètres
    E = 1.1e11
    S = 50 * 20 * 1e-6
    L = 0.2
    rho = 4430.0
    rot_max = 39000.0
    B = 7.7e8
    m = 0.557
    sigma_y = 9.55e8
    newt_initial_deltap = 1.0e-5
    newt_iter_max = 1000
    newt_tolerence = 1e-3
    Nelem = 10
    loading_steps = 20


    # Déinition de la pâle et du maillage
    pale = Pale(E, S, L, rho, rot_max, B, m, sigma_y)
    mesh = Mesh(Nelem, L)

    # Retour radial (étape locale)
    rr = ReturnMapping(pale, newt_initial_deltap, newt_iter_max, newt_tolerence)
    rr.plot_sigma_eps(1000)

    # Solver (étape globale)
    try:
        solution = PaleSolver(1, loading_steps, newt_iter_max, newt_tolerence, pale, mesh, rr)
        solution.loadingLoop()
        solution.plot_results(1)
    except Exception as e:
        print(f"Exception: {e}")
