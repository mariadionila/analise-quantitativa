# 📈 DioInvest AI - Sistema Inteligente de Análise Quantitativa de Ativos

Este projeto implementa um **Web App Interativo em Streamlit** para apoio à tomada de decisão em investimentos financeiros utilizando técnicas de **Econometria**, **Machine Learning** e **Deep Learning**.

O objetivo é transformar um simples visualizador de dados em um sistema inteligente capaz de analisar ativos, modelar riscos, prever retornos futuros e, futuramente, otimizar carteiras de investimento.

O projeto foi desenvolvido para a disciplina **Tópicos Avançados em Finanças – Ciência de Dados para Negócios (UFPB)**.

---

# 🚀 Funcionalidades

O usuário escolhe um mercado, seleciona os ativos e define o período da análise. O aplicativo realiza automaticamente o download dos dados históricos e organiza os resultados em módulos independentes.

## 📊 Visão Geral

- Evolução dos preços ajustados
- Retorno acumulado
- Retorno anualizado
- Volatilidade anualizada
- Drawdown máximo
- Estatísticas descritivas

---

## 📈 CAPM

Implementação completa do Capital Asset Pricing Model.

São estimados:

- Beta
- Alfa
- Retorno esperado
- Prêmio de risco
- Coeficiente de determinação (R²)
- p-value do Beta
- Gráfico de regressão

---

## 📉 Modelo de Três Fatores (Fama-French)

Decomposição dos retornos utilizando:

- Fator de Mercado (MKT-RF)
- SMB (Small Minus Big)
- HML (High Minus Low)

O módulo apresenta:

- Coeficientes dos fatores
- Significância estatística
- R²
- Comparação entre ativos

---

## 📊 Modelos ARCH/GARCH

Análise da dinâmica da volatilidade.

O sistema estima:

- ARCH
- GARCH(1,1)
- Persistência da volatilidade
- Volatilidade condicional
- Previsão de volatilidade futura

---

# 🤖 Inteligência Artificial

O aplicativo possui um módulo completo de previsão de retornos utilizando algoritmos de aprendizado de máquina.

Modelos implementados:

- Random Forest Regressor
- XGBoost
- LightGBM

Para cada modelo são exibidos:

- Previsões
- MAE
- RMSE
- R²
- Comparação entre valores reais e previstos

---

# 🧠 Deep Learning

Também foram implementadas Redes Neurais Recorrentes para séries temporais.

Modelos disponíveis:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

Os modelos são treinados utilizando janelas temporais dos preços históricos e exibem:

- Curvas de treinamento
- Loss
- Predições
- Comparação entre previsão e valores reais

---

# 📋 Validação

Para evitar vazamento temporal dos dados, o projeto utiliza técnicas específicas para séries financeiras.

Recursos implementados:

- Time Series Cross Validation
- Divisão cronológica dos dados
- Métricas de desempenho
- Comparação entre modelos

---

# 📈 Backtesting

O módulo de Backtesting permite avaliar o desempenho histórico das estratégias geradas pelos modelos.

São apresentados:

- Retorno acumulado
- Comparação Buy & Hold
- Curva de patrimônio
- Estatísticas da estratégia

---

# 📂 Fontes de Dados

Os dados são obtidos automaticamente através das seguintes fontes:

- Yahoo Finance (yfinance)
- Kenneth French Data Library
- pandas-datareader

Caso alguma fonte esteja indisponível, o sistema permite importar arquivos CSV contendo preços históricos e fatores de Fama-French.

O projeto utiliza exclusivamente dados reais.

---

# ▶️ Como Executar

Clone o repositório:

```bash
git clone https://github.com/mariadionila/analise-quantitativa.git
```

Entre na pasta:

```bash
cd analise-quantitativa
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente.

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o aplicativo:

```bash
streamlit run app.py
```

Acesse:

```
http://localhost:8501
```

---

# 💻 Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Statsmodels
- ARCH
- yfinance
- pandas-datareader
- scikit-learn
- XGBoost
- LightGBM
- TensorFlow / Keras
- Plotly
- Matplotlib

---

# 📁 Estrutura do Projeto

```
analise-quantitativa/

│
├── app.py
├── requirements.txt
├── README.md
│
├── pages/
│   ├── Visao_Geral.py
│   ├── CAPM.py
│   ├── CAPM_e_Fama_French.py
│   ├── GARCH.py
│   ├── Predicao_IA.py
│   ├── DeepLearning.py
│   ├── Validacao.py
│   └── Backtesting.py
│
├── utils/
├── data/
└── assets/
```

---

# 📊 Metodologia

O fluxo do sistema segue três etapas principais:

```
Dados Históricos
        │
        ▼
Econometria
(CAPM • Fama-French • GARCH)
        │
        ▼
Machine Learning
(Random Forest • XGBoost • LightGBM)
        │
        ▼
Deep Learning
(LSTM • GRU)
        │
        ▼
Validação Temporal
        │
        ▼
Backtesting
```

Cada módulo fornece informações complementares para apoiar decisões de investimento baseadas em métodos quantitativos.

---

# 🔄 Próximas Implementações

A próxima etapa do projeto contempla:

- Otimização de Carteiras (Markowitz)
- Fronteira Eficiente
- Black-Litterman
- Hierarchical Risk Parity (HRP)
- Riskfolio-Lib
- Sistema completo de recomendação de portfólios

---

# 👨‍💻 Autor

**Maria Dionila**

Graduando em  Ciências de Dados para Negocios - Ufpb

Projeto desenvolvido para a disciplina **Tópicos Avançados em Finanças** da Universidade Federal da Paraíba (UFPB).

---


Este projeto foi desenvolvido para fins acadêmicos.
