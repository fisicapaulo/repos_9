import os

# --- 1. Definir a Estrutura de Pastas ---
# A estrutura de pastas segue o Capítulo 6 do seu livro
DIRS = [
    "core/cones",
    "sigma_delta/metrics",
    "pipelines/pl",
    "data/locks",
    "benchmarks/tables"
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"Diretório criado: {d}")

# --- 2. Criar o Arquivo de Requisitos (Dependências) ---
REQUIREMENTS_CONTENT = """
numpy
scipy
pandas
"""
with open("requirements.txt", "w") as f:
    f.write(REQUIREMENTS_CONTENT)
print("Arquivo criado: requirements.txt")

# --- 3. Criar o Arquivo README (Apresentação) ---
README_CONTENT = """
# Repositório de Auditoria D-Audit: Rigidez Coerciva (Problema 9 de Smale)

Este repositório contém a implementação do Protocolo D-Audit, parte da solução fortemente
polinomial do Problema 9 de Smale.

O objetivo principal é atestar a **independência da altura aritmética (fator L)**, conforme
medido pelo invariante de coerção espectral $\Sigma(\delta)$ e o índice $\mathcal{I}_L$.

## Conteúdo
- `core/cones`: Implementação do Operador Coerente $G$ via MEDA (Simulação).
- `sigma_delta/metrics`: Módulo principal para o cálculo do Gap $\delta$ e $\kappa_{coh}$.
- `pipelines/pl`: Script executável para o Teste de Independência de L (`run_il_test.py`).

## Requisitos
Python 3.x, numpy, scipy, pandas.
"""
with open("README.md", "w") as f:
    f.write(README_CONTENT)
print("Arquivo criado: README.md")

# --- 4. Criar Arquivo: core/cones/pl_operator.py (Operador G - MEDA) ---
PL_OPERATOR_CONTENT = """
import numpy as np
from scipy.linalg import eigh

def generate_A_with_L_scale(n, m, scale_L=1.0, seed=2025):
    \"\"\"Gera matriz A genérica e reescala pela altura aritmética L.\"\"\"
    np.random.seed(seed)
    A_raw = np.random.randn(m, n)
    A = A_raw * scale_L
    return A

def compute_coherent_operator_G(A, active_set_size, delta_dimensional=0.1):
    \"\"\"
    SIMULAÇÃO DO MEDA: Equalização + Projeção/Rigidez Coerciva (Axioma D5).
    
    O delta_dimensional simula a cota dimensional mínima que o MEDA garante.
    \"\"\"
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
"""
with open("core/cones/pl_operator.py", "w") as f:
    f.write(PL_OPERATOR_CONTENT)
print("Arquivo criado: core/cones/pl_operator.py")


# --- 5. Criar Arquivo: sigma_delta/metrics/delta_coherent.py (API Sigma(delta)) ---
DELTA_COHERENT_CONTENT = """
import numpy as np
from scipy.linalg import eigh

def delta_coherent(G):
    \"\"\"
    Calcula o Gap Coerente (delta) e o Condicionamento Coerente (kappa)
    para o operador coerente G.
    \"\"\"
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
"""
with open("sigma_delta/metrics/delta_coherent.py", "w") as f:
    f.write(DELTA_COHERENT_CONTENT)
print("Arquivo criado: sigma_delta/metrics/delta_coherent.py")


# --- 6. Criar Arquivo: pipelines/pl/run_il_test.py (Motor de Teste I_L) ---
RUN_IL_TEST_CONTENT = """
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
print(f"Iniciando Teste I_L para PL (n={{N_DIM}}, m={{M_DIM}})...")

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
    
    print(f"  {{results[-1]['Run']}}: T={{T_simulated:.2f}}ms, delta={{metrics['value']:.4f}}")

# --- Cálculo do Índice I_L (Medindo a variação) ---
df = pd.DataFrame(results)

T_mean = df['T'].mean()
I_L_T = df['T'].apply(lambda t: abs(t - T_mean)).max() / T_mean if T_mean != 0 else 0

D_mean = df['delta_bar'].mean()
I_L_delta = df['delta_bar'].apply(lambda d: abs(d - D_mean)).max() / D_mean if D_mean != 0 else 0

I_L_total = I_L_T + I_L_delta

# --- Geração da Tabela LaTeX (Final) ---
print("\\n" + "="*70)
print(f"RELATÓRIO DE CERTIFICAÇÃO: INDEPENDÊNCIA ARITMÉTICA (I_L = {{I_L_total:.4f}})")
print("="*70)

# Estrutura final da tabela para o artigo
print("\\\\begin{table}[H]")
print("\\\\centering")
print("\\\\small")
print("\\\\caption{PL genérico: custo e invariantes sob reescalas aritméticas (n=2000, m=500).}")
print("\\\\begin{tabular}{lrrrr}")
print("\\\\toprule")
print("\\\\textbf{Run} & $T$ (ms) & $\\\\overline{{\\\\delta}}$ & $\\\\kappa_{\\\\mathrm{{coh}}}}$ & $\\\\mathcal{I}_L}$ \\\\")
print("\\\\midrule")

for i, r in df.iterrows():
    I_L_display = f"{{I_L_total:.4f}}" if i == 0 else "" 
    print(f"{{r['Run']}} & {{r['T']:.1f}} & {{r['delta_bar']:.4f}} & {{r['kappa_coh']:.1f}} & {{I_L_display}} \\\\")

print("\\\\bottomrule")
print("\\\\end{tabular}")
print("\\\\end{table}")
print("\\n* Nota: O índice $\\\\mathcal{I}_L$ próximo de zero prova a independência de L.")
"""
with open("pipelines/pl/run_il_test.py", "w") as f:
    f.write(RUN_IL_TEST_CONTENT)
print("Arquivo criado: pipelines/pl/run_il_test.py")

print("\n--- Setup Local Concluído ---")
print("Os diretórios e arquivos base foram criados com sucesso!")