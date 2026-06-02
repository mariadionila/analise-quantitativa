# modelos arch/garch

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model

# configuração da página

st.set_page_config(
    page_title="ARCH/GARCH",
    layout="wide"
)

# título

st.title("📉 Modelos ARCH/GARCH")

st.markdown("""

## Modelagem da Heterocedasticidade Condicional

Os modelos ARCH/GARCH são utilizados para
analisar a dinâmica da volatilidade dos
ativos financeiros.

Esses modelos conseguem capturar:

- agrupamento de volatilidade
- persistência do risco
- períodos turbulentos
- mudanças dinâmicas na variância

""")

# sidebar

st.sidebar.header("⚙️ Configurações")

# lista de ativos

lista_ativos = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
    "WEGE3.SA",
    "BBAS3.SA",
    "MGLU3.SA",
    "SUZB3.SA",
    "B3SA3.SA",
    "RENT3.SA",
    "LREN3.SA"
]

# seleção dos ativos

ativos = st.sidebar.multiselect(
    "Selecione os ativos",
    lista_ativos,
    default=[
        "PETR4.SA",
        "VALE3.SA",
        "ITUB4.SA"
    ]
)

# período

data_inicio = st.sidebar.date_input(
    "Data Inicial",
    pd.to_datetime("2020-01-01")
)

data_fim = st.sidebar.date_input(
    "Data Final",
    pd.to_datetime("today")
)

# botão

executar = st.sidebar.button(
    "🚀 Executar Análise"
)

# execução

if executar:

    resultados_garch = []

    # loop dos ativos

    for ativo in ativos:

        try:

            st.header(f"📊 {ativo}")

            # download dos dados

            dados_ativo = yf.download(
                ativo,
                start=data_inicio,
                end=data_fim,
                progress=False
            )

            # validar dados

            if dados_ativo.empty:

                st.warning(
                    f"Sem dados para {ativo}"
                )

                continue

            # retornos

            retorno = (
                dados_ativo["Close"]
                .pct_change()
                .dropna()
            )

            # validar retornos

            if len(retorno) < 30:

                st.warning(
                    f"Poucos dados para GARCH em {ativo}"
                )

                continue

            # modelo garch(1,1)

            modelo_garch = arch_model(
                retorno * 100,
                vol="Garch",
                p=1,
                q=1
            )

            resultado = modelo_garch.fit(
                disp="off"
            )

            # volatilidade condicional

            volatilidade = (
                resultado.conditional_volatility
            )

            # previsão

            previsao = resultado.forecast(
                horizon=5
            )

            volatilidade_prevista = (
                previsao.variance
                .iloc[-1]
                .mean()
            )

            # parâmetros

            omega = resultado.params["omega"]

            alpha = resultado.params["alpha[1]"]

            beta = resultado.params["beta[1]"]

            # resultados

            resultados_garch.append({

                "Ativo": ativo,

                "Omega": round(
                    omega,
                    6
                ),

                "Alpha": round(
                    alpha,
                    4
                ),

                "Beta": round(
                    beta,
                    4
                ),

                "Volatilidade Prevista": round(
                    volatilidade_prevista,
                    4
                )

            })

            # gráfico volatilidade

            st.subheader(
                "📈 Volatilidade Condicional"
            )

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.plot(
                volatilidade,
                label="Volatilidade"
            )

            ax.set_title(
                f"GARCH(1,1) - {ativo}"
            )

            ax.set_xlabel(
                "Tempo"
            )

            ax.set_ylabel(
                "Volatilidade"
            )

            ax.legend()

            st.pyplot(fig)

        except Exception as e:

            st.error(
                f"Erro GARCH ({ativo}): {e}"
            )

    # resultados finais

    if len(resultados_garch) > 0:

        df_garch = pd.DataFrame(
            resultados_garch
        )

        st.header(
            "📊 Resultado Final GARCH"
        )

        st.dataframe(
            df_garch,
            use_container_width=True
        )

        # gráfico comparativo

        st.subheader(
            "📉 Comparação de Volatilidade"
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 5)
        )

        ax2.bar(
            df_garch["Ativo"],
            df_garch["Volatilidade Prevista"]
        )

        ax2.set_title(
            "Volatilidade Prevista"
        )

        ax2.set_ylabel(
            "Volatilidade"
        )

        st.pyplot(fig2)

    else:

        st.warning(
            "Nenhum ativo válido para análise."
        )
