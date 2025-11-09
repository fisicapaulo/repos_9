
import numpy as np
from scipy.linalg import eigh

def delta_coherent(G):
    """
    Calcula o Gap Coerente (delta) e o Condicionamento Coerente (kappa)
    para o operador coerente G.
    """
    # Usa eigh para autovalores de matrizes simétricas
    eigenvalues = eigh(G, eigvals_only=True)
    
    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]
    
    delta = lambda_min / lambda_max
    kappa = 1.0 / delta
    
    return {
        "value": delta,
        "kappa_coh": kappa
    }
