
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
