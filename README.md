# 📈 DioInvest AI
### Sistema Inteligente de Análise Quantitativa e Otimização de Carteiras

> Plataforma desenvolvida em **Python** e **Streamlit** para apoio à tomada de decisão em investimentos, integrando modelos de **Econometria**, **Machine Learning**, **Deep Learning** e **Teoria Moderna de Carteiras (Markowitz)**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![License](https://img.shields.io/badge/License-Acadêmico-green)

---

# 📌 Sobre o Projeto

O **DioInvest AI** foi desenvolvido como projeto final da disciplina **Tópicos Avançados em Finanças**, do curso de Ciência de Dados para Negócios da Universidade Federal da Paraíba (UFPB).

O sistema foi criado para automatizar todo o processo de análise quantitativa de investimentos, permitindo que o usuário avalie ativos financeiros desde a coleta dos dados históricos até a construção de uma carteira otimizada.

O principal objetivo é fornecer uma plataforma capaz de integrar técnicas econométricas, modelos de Inteligência Artificial e otimização de carteiras em um único fluxo de apoio à tomada de decisão.

---

# 🎯 Objetivos

O sistema permite que o investidor:

- Avalie o risco dos ativos financeiros;
- Analise fatores de mercado através de modelos econométricos;
- Preveja retornos utilizando Machine Learning e Deep Learning;
- Valide a qualidade dos modelos preditivos;
- Simule estratégias através de Backtesting;
- Construa uma carteira ótima utilizando o Modelo Média-Variância de Markowitz.

---

# 🏗 Arquitetura do Sistema

O projeto foi dividido em três grandes camadas, seguindo exatamente o fluxo de tomada de decisão utilizado na análise quantitativa.

```text
Coleta de Dados
        │
        ▼
Análise Exploratória
        │
        ▼
Fase I
Modelagem do Risco
(CAPM • Fama-French • ARCH/GARCH)
        │
        ▼
Fase II
Machine Learning
(Random Forest • XGBoost • LightGBM)
        │
        ▼
Deep Learning
(LSTM • GRU)
        │
        ▼
Validação Temporal
(TimeSeriesSplit)
        │
        ▼
Backtesting
        │
        ▼
Fase III
Otimização de Carteiras
(Markowitz)
        │
        ▼
Carteira Recomendada
```

Cada módulo da aplicação alimenta o módulo seguinte, permitindo que a carteira final seja construída utilizando as previsões obtidas pelos modelos desenvolvidos.

---

# 🚀 Funcionalidades

## 📊 Coleta de Dados

O sistema realiza automaticamente a obtenção de dados financeiros através da biblioteca **yfinance**, permitindo ao usuário selecionar ativos da B3 e definir o período de análise.

São coletados:

- preços históricos;
- retornos diários;
- volume negociado;
- estatísticas básicas.

---

## 📈 Análise Exploratória

Nesta etapa são apresentadas diversas análises estatísticas dos ativos financeiros.

Entre elas:

- Retorno acumulado;
- Retorno diário;
- Volatilidade;
- Correlação;
- Drawdown;
- Estatísticas descritivas;
- Visualização gráfica dos preços.

---

# 📉 Fase I — Modelagem do Risco

A primeira camada do sistema é responsável pela avaliação do risco financeiro dos ativos.

São utilizados modelos clássicos de econometria.

## CAPM

O modelo Capital Asset Pricing Model (CAPM) estima o retorno esperado de cada ativo considerando o risco sistemático.

São apresentados:

- Beta;
- Alfa;
- Prêmio de risco;
- Retorno esperado;
- Regressão Linear;
- Gráficos comparativos.

---

## Modelo Fama-French

O modelo de três fatores amplia o CAPM adicionando fatores relacionados ao tamanho e ao valor das empresas.

São analisados:

- MKT-RF;
- SMB;
- HML;
- Coeficientes;
- Significância estatística;
- Regressão múltipla.

---

## ARCH/GARCH

Os modelos ARCH e GARCH são utilizados para modelar a volatilidade condicional dos ativos.

São apresentados:

- Volatilidade condicional;
- Persistência da volatilidade;
- Previsão de volatilidade futura;
- Gráficos da variância condicional.

---

# 🤖 Fase II — Machine Learning

Após o filtro de risco, os ativos passam pelo módulo de Inteligência Artificial.

Foram implementados diversos algoritmos supervisionados para previsão dos retornos.

## Modelos utilizados

- Random Forest
- XGBoost
- LightGBM

Cada modelo é treinado utilizando séries temporais e posteriormente comparado através de métricas de desempenho.

---

## Métricas avaliadas

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Essas métricas permitem identificar qual algoritmo apresentou melhor capacidade preditiva.

---

# 🧠 Deep Learning

Além dos modelos tradicionais, o sistema também implementa Redes Neurais Artificiais para captura de padrões temporais complexos.

Modelos implementados:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

Durante o treinamento são apresentados:

- Curva de Loss;
- Curva de Validação;
- Comparação entre valores reais e previstos.

---

# ✅ Validação dos Modelos

Para evitar problemas de **Data Leakage**, foi utilizada validação específica para séries temporais.

Foram implementadas técnicas como:

- Time Series Cross Validation;
- Separação cronológica entre treino, validação e teste;
- Comparação entre modelos.

Essa abordagem garante maior confiabilidade aos resultados obtidos.

---

# 📈 Backtesting

O módulo de Backtesting permite simular o comportamento histórico da estratégia desenvolvida.

São comparadas diferentes abordagens de investimento.

Entre elas:

- Estratégia baseada nas previsões do modelo;
- Buy and Hold;
- Evolução do patrimônio;
- Retorno acumulado.

Esse módulo permite avaliar se a estratégia realmente agregaria valor ao investidor.

---

# 💼 Fase III — Otimização de Carteiras

Após a previsão dos retornos, os valores previstos alimentam o modelo de otimização.

Foi implementada a Teoria Moderna de Portfólios de Markowitz.

O sistema calcula automaticamente:

- Retorno esperado da carteira;
- Risco da carteira;
- Matriz de covariância;
- Fronteira eficiente;
- Carteira de mínima variância;
- Carteira de maior Índice de Sharpe;
- Pesos ótimos para cada ativo.

Dessa forma, o investidor obtém uma carteira otimizada considerando simultaneamente retorno esperado e risco.

---

# 🔄 Fluxo Completo da Aplicação

O funcionamento completo do sistema segue o fluxo abaixo.

```text
Escolha dos Ativos
        │
        ▼
Download Automático
        │
        ▼
Análise Exploratória
        │
        ▼
CAPM
        │
        ▼
Fama-French
        │
        ▼
ARCH/GARCH
        │
        ▼
Machine Learning
        │
        ▼
Deep Learning
        │
        ▼
Validação
        │
        ▼
Backtesting
        │
        ▼
Markowitz
        │
        ▼
Carteira Final Recomendada
```

Todo esse processo ocorre automaticamente dentro da plataforma.

---

# 📂 Estrutura do Projeto

```text
analise-quantitativa/

│── app.py
│── requirements.txt
│── README.md
│
├── pages/
│   ├── Analise_Exploratoria.py
│   ├── CAPM.py
│   ├── Fama_French.py
│   ├── ARCH_GARCH.py
│   ├── Predicao_IA.py
│   ├── DeepLearning.py
│   ├── Validacao.py
│   ├── Backtesting.py
│   └── Otimizacao_Carteira.py
│
├── models/
│
├── data/
│
├── utils/
│
└── assets/
```

---

# 💻 Tecnologias Utilizadas

Linguagem:

- Python 3.12

Framework Web:

- Streamlit

Bibliotecas:

- Pandas
- NumPy
- Scikit-Learn
- Statsmodels
- ARCH
- TensorFlow
- Keras
- XGBoost
- LightGBM
- Plotly
- Matplotlib
- Seaborn
- yfinance
- CVXPY

---

# ⚙️ Instalação

Clone o projeto

```bash
git clone https://github.com/SEU-USUARIO/analise-quantitativa.git
```

Entre na pasta

```bash
cd analise-quantitativa
```

Crie um ambiente virtual

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / MacOS

```bash
source .venv/bin/activate
```

Instale as dependências

```bash
pip install -r requirements.txt
```

---

# ▶️ Executando o Projeto

Execute:

```bash
streamlit run app.py
```

Após a inicialização, a aplicação abrirá automaticamente no navegador.

---

# 📸 Demonstração

Adicione aqui imagens da aplicação.

Exemplo:

```
assets/home.png

assets/capm.png

assets/predicao.png

assets/markowitz.png
```

Ou GIFs demonstrando o funcionamento do sistema.

---

# 📊 Resultados Esperados

O sistema permite:

- avaliar o risco dos ativos;
- identificar empresas com maior potencial de retorno;
- prever preços utilizando Inteligência Artificial;
- validar os modelos;
- comparar estratégias;
- construir uma carteira ótima.

Todo esse fluxo ocorre automaticamente em uma única plataforma.

---

# 📚 Metodologia

O projeto foi desenvolvido utilizando uma arquitetura composta por três fases principais.

### Fase I

Avaliação do risco financeiro através dos modelos:

- CAPM
- Fama-French
- ARCH/GARCH

### Fase II

Predição dos retornos utilizando:

- Machine Learning
- Deep Learning

### Fase III

Otimização utilizando:

- Modelo Média-Variância de Markowitz

Cada etapa fornece informações para a etapa seguinte, permitindo uma integração completa entre risco, previsão e otimização.

---

# 🔮 Trabalhos Futuros

Como possíveis evoluções do projeto, destacam-se:

- Inclusão de novos modelos econométricos;
- Implementação do modelo Black-Litterman;
- Implementação do Hierarchical Risk Parity (HRP);
- Uso de dados fundamentalistas;
- Inclusão de indicadores macroeconômicos;
- Rebalanceamento automático da carteira;
- Explicabilidade utilizando SHAP;
- Publicação da aplicação em ambiente de produção.

---

# 👩‍💻 Autora

**Maria Dionila**

Universidade Federal da Paraíba (UFPB)

Curso de Ciência de Dados para Negócios

Disciplina: **Tópicos Avançados em Finanças**

---

# 📄 Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos como requisito da disciplina **Tópicos Avançados em Finanças**, da Universidade Federal da Paraíba.
