
import numpy as np
from scipy.linalg import eigh

def generate_A_with_L_scale(n, m, scale_L=1.0, seed=2025):
    """Gera matriz A genérica e reescala pela altura aritmética L."""
    np.random.seed(seed)
    A_raw = np.random.randn(m, n)
    A = A_raw * scale_L
    return A

def compute_coherent_operator_G(A, active_set_size, delta_dimensional=0.1):
    """
    SIMULAÇÃO DO MEDA: Equalização + Projeção/Rigidez Coerciva (Axioma D5).
    
    O delta_dimensional simula a cota dimensional mínima que o MEDA garante.
    """
    r = active_set_size
    A_active = A[:, :r]
    H_base = A_active.T @ A_active
    
    # 3. Injeção de Rigidez Coerciva (Axioma D5)
    n_reduced = H_base.shape[0]
    D_coercive = np.eye(n_reduced) * delta_dimensional * np.mean(np.diag(H_base))
    
    G = H_base + D_coercive
    
    # 4. Normalização simples
    G = G / np.mean(np.diag(G))
    
    return G
