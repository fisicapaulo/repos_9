# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.cones.pl_operator import compute_coherent_operator_G, generate_A_with_L_scale
from sigma_delta.metrics.delta_coherent import delta_coherent
import time
import numpy as np
import pandas as pd

# --- Parâmetros ---
N_DIM = 2000
M_DIM = 500
R_RANK = 400
DELTA_DIM = 0.1
L_RUNS = [1.0, 1e6, 1e12] # Reescalas L para testar independencia

results = []
print(f"Iniciando Teste I_L para PL (n={N_DIM}, m={M_DIM})...")

for L_scale in L_RUNS:
    A = generate_A_with_L_scale(N_DIM, M_DIM, scale_L=L_scale, seed=2025)
    start_time = time.time()
    G = compute_coherent_operator_G(A, R_RANK, delta_dimensional=DELTA_DIM)
    end_time = time.time()
    
    metrics = delta_coherent(G)
    T_simulated = (end_time - start_time) * 1000 
    
    results.append({
        "Run": f"L-{L_scale:.0e}",
        "T": T_simulated,
        "delta_bar": metrics['value'],
        "kappa_coh": metrics['kappa_coh']
    })
    
    # Linha A (Corrigida)
    print("  {}: T={:.2f}ms, delta={:.4f}".format(results[-1]['Run'], T_simulated, metrics['value']))

# --- Cálculo do Índice I_L (Medindo a variação) ---
df = pd.DataFrame(results)

T_mean = df['T'].mean()
I_L_T = df['T'].apply(lambda t: abs(t - T_mean)).max() / T_mean if T_mean != 0 else 0

D_mean = df['delta_bar'].mean()
I_L_delta = df['delta_bar'].apply(lambda d: abs(d - D_mean)).max() / D_mean if D_mean != 0 else 0

I_L_total = I_L_T + I_L_delta

# --- Geração da Tabela LaTeX (Final) ---
print("\n" + "="*70)
# Linha B (Corrigida)
print("RELATÓRIO DE CERTIFICAÇÃO: INDEPENDÊNCIA ARITMÉTICA (I_L = {:.4f})".format(I_L_total))
print("="*70)

print("\\begin{table}[H]")
print("\\centering")
print("\\small")
print("\\caption{PL genérico: custo e invariantes sob reescalas aritméticas (n=2000, m=500).}")
print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("\\textbf{Run} & $T$ (ms) & $\\overline{\\delta}$ & $\\kappa_{\\mathrm{coh}}$ & $\\mathcal{I}_L$ \\\\")
print("\\midrule")

for i, r in df.iterrows():
    I_L_display = "{:.4f}".format(I_L_total) if i == 0 else "" 
    # Linha C (Corrigida)
    print("{Run} & {T:.1f} & {delta_bar:.4f} & {kappa_coh:.1f} & {I_L_display} \\\\".format(Run=r['Run'], T=r['T'], delta_bar=r['delta_bar'], kappa_coh=r['kappa_coh'], I_L_display=I_L_display))

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print("\n* Nota: O índice $\\mathcal{I}_L$ próximo de zero prova a independência de L.")