# CAPM_e_Fama_French.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Seleção de Ativos - CAPM e Fama-French",
    layout="wide"
)

st.title("Aplicação dos CAPM")
st.markdown("## Análise CAPM dos Ativos Selecionados")

st.sidebar.header("Parâmetros da Análise")

lista_ativos = [
    "PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA",
    "WEGE3.SA","BBAS3.SA","MGLU3.SA","SUZB3.SA","JBSS3.SA",
    "RENT3.SA","LREN3.SA"
]

ativos = st.sidebar.multiselect(
    "Selecione os ativos",
    lista_ativos,
    default=["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA","WEGE3.SA","BBAS3.SA"]
)

benchmark="^BVSP"

data_inicio=st.sidebar.date_input("Data Inicial",pd.to_datetime("2023-01-01"))
data_fim=st.sidebar.date_input("Data Final",pd.to_datetime("today"))
taxa_livre=st.sidebar.number_input("Taxa Livre de Risco (%)",value=10.75)/100

st.sidebar.markdown("## Filtros")
beta_min=st.sidebar.number_input("Beta mínimo",value=-5.0,step=0.1)
beta_max=st.sidebar.number_input("Beta máximo",value=5.0,step=0.1)
alpha_opcao=st.sidebar.selectbox("Filtrar Alpha",["Todos","Positivo","Negativo"])

st.sidebar.markdown("### Ativos analisados")
for a in ativos:
    st.sidebar.write(a)

if st.sidebar.button("Executar Análise CAPM"):

    dados_benchmark=yf.download(benchmark,start=data_inicio,end=data_fim,progress=False)
    retorno_mercado=dados_benchmark["Close"].pct_change().dropna()

    resultados=[]

    for ativo in ativos:
        try:
            dados=yf.download(ativo,start=data_inicio,end=data_fim,progress=False)
            retorno=dados["Close"].pct_change().dropna()

            df=pd.concat([retorno,retorno_mercado],axis=1).dropna()
            df.columns=["Ativo","Mercado"]

            X=sm.add_constant(df["Mercado"])
            y=df["Ativo"]

            modelo=sm.OLS(y,X).fit()

            alpha=float(modelo.params["const"])
            beta=float(modelo.params["Mercado"])

            retorno_medio=df["Mercado"].mean()*252
            retorno_esperado=taxa_livre+beta*(retorno_medio-taxa_livre)
            premio=retorno_esperado-taxa_livre
            volatilidade=df["Ativo"].std()*np.sqrt(252)

            resultados.append({
                "Ativo":ativo,
                "Alpha":round(alpha,6),
                "Beta":round(beta,4),
                "Retorno Esperado (%)":round(retorno_esperado*100,2),
                "Prêmio de Risco (%)":round(premio*100,2),
                "Volatilidade (%)":round(volatilidade*100,2),
                "R²":round(modelo.rsquared,4)
            })

        except Exception as e:
            st.error(f"Erro ao processar {ativo}: {e}")

    df_resultados=pd.DataFrame(resultados)

    if not df_resultados.empty:
        df_resultados=df_resultados[
            (df_resultados["Beta"]>=beta_min)&
            (df_resultados["Beta"]<=beta_max)
        ]

        if alpha_opcao=="Positivo":
            df_resultados=df_resultados[df_resultados["Alpha"]>0]
        elif alpha_opcao=="Negativo":
            df_resultados=df_resultados[df_resultados["Alpha"]<0]

    if df_resultados.empty:
        st.warning("Nenhum ativo encontrado para os filtros selecionados.")
        st.stop()

    df_resultados=df_resultados.sort_values(
        by="Prêmio de Risco (%)",
        ascending=False
    )

    st.subheader("Ranking de Ativos")
    st.dataframe(
        df_resultados.style.background_gradient(
            cmap="Blues",
            subset=["Retorno Esperado (%)","Prêmio de Risco (%)"]
        ),
        use_container_width=True
    )

    st.subheader("Resumo Estatístico")
    st.dataframe(df_resultados.describe(),use_container_width=True)

    c1,c2,c3=st.columns(3)
    c1.metric("Maior Beta",df_resultados.loc[df_resultados["Beta"].idxmax(),"Ativo"])
    c2.metric("Menor Beta",df_resultados.loc[df_resultados["Beta"].idxmin(),"Ativo"])
    c3.metric("Maior Prêmio",df_resultados.loc[df_resultados["Prêmio de Risco (%)"].idxmax(),"Ativo"])

    st.subheader("Beta dos Ativos")
    fig,ax=plt.subplots(figsize=(10,5))
    cores=["green" if b>=1 else "steelblue" for b in df_resultados["Beta"]]
    ax.bar(df_resultados["Ativo"],df_resultados["Beta"],color=cores)
    ax.axhline(1,color="red",linestyle="--")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Retorno Esperado")
    fig2,ax2=plt.subplots(figsize=(10,5))
    ax2.bar(df_resultados["Ativo"],df_resultados["Retorno Esperado (%)"],color="royalblue")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
