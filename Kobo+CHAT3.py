import streamlit as st
import pandas as pd
import requests
import json
import openai
import matplotlib.pyplot as plt
import io
import contextlib
import os

import unicodedata
import re

def normalizar_colunas(df):
    df.columns = [
        re.sub(r"[“”\"']", "", unicodedata.normalize("NFKC", c)).strip()
        for c in df.columns
    ]
    return df

st.set_page_config(page_title="Kobo Dashboard + ChatGPT", layout="wide")

st.title("📊 KoboToolbox Data Analyzer + ChatGPT 📡")

if "nome_utilizador" not in st.session_state:
    st.session_state.nome_utilizador = st.text_input("👤 Introduza o seu nome para começar:")
    if not st.session_state.nome_utilizador:
        st.stop()

token = st.sidebar.text_input("🔐 API Token do KoboToolbox", type="password")
form_id = st.sidebar.text_input("🆔 UID do Formulário KoboToolbox (ex: aAQqHmPbMUaLAehcFQNpTi)")
openai_api_key = st.sidebar.text_input("🔐 OpenAI API Key", type="password")

modelo_openai = st.sidebar.selectbox("🤖 Modelo GPT", ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"], index=0)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": (
                f"O utilizador chama-se {st.session_state.nome_utilizador}. "
                "Vais analisar dados em CSV. Responde em português de forma clara. "
                "Usa **exclusivamente** os nomes das colunas fornecidas, tal como aparecem. "
                "Nunca inventes nomes de colunas. "
                "Se tiveres de fazer contas, mostra os passos. Não inventes valores. "
                "Evita estimativas e, se não conseguires calcular com precisão, diz isso explicitamente. "
                "Se geraste código Python para gráficos, escreve-o como bloco de código com Python para ser identificado. "
                "Sempre que escrevas código, não acabes a meio, certifica-te de que o código está completo e funcional. "
                "Se precisares de gerar gráficos, certifica-te de que os dados estão prontos e que o gráfico é coerente com os dados fornecidos. "
                "Usa sempre `fig, ax = plt.subplots()` e termina com `st.pyplot(fig)` para garantir compatibilidade com Streamlit. "
                "Nunca uses `plt.show()` nem `st.pyplot()` sem argumento. "
                "Os dados já estão disponíveis no ficheiro 'dados_para_analise.csv'. Usa `df = pd.read_csv('dados_para_analise.csv')` e os nomes de colunas exatos."
            )
        }
    ]

def expandir_colunas_dict(df):
    novas_colunas = []
    for col in df.columns:
        if col == "_attachments":
            continue
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            linhas_validas = df[col].apply(lambda x: isinstance(x, dict))
            expandidas = df.loc[linhas_validas, col].apply(pd.Series)
            expandidas = expandidas.add_prefix(f"{col}/")
            df = df.drop(columns=[col])
            df = pd.concat([df, expandidas], axis=1)
            novas_colunas.extend(expandidas.columns)
    return df

def expandir_listas_dict(df):
    for col in df.columns:
        if col == "_attachments":
            continue
        if df[col].apply(lambda x: isinstance(x, list) and all(isinstance(i, dict) for i in x)).any():
            df = df.explode(col).reset_index(drop=True)
            dict_expandidos = pd.json_normalize(df[col]).add_prefix(f"{col}/")
            df = df.drop(columns=[col]).reset_index(drop=True)
            dict_expandidos = dict_expandidos.reset_index(drop=True)
            df = pd.concat([df, dict_expandidos], axis=1)
    return df

if token and form_id:
    url = f"https://kf.kobotoolbox.org/api/v2/assets/{form_id}/data/?format=json"
    headers = {"Authorization": f"Token {token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()["results"]
        df = pd.DataFrame(data)

        if "_attachments" in df.columns:
            df = df.drop(columns=["_attachments"])

        df = expandir_listas_dict(df)
        df = expandir_colunas_dict(df)

        colunas_validas = [col for col in df.columns if df[col].map(type).isin([str, int, float, bool, pd.Timestamp]).all()]
        if colunas_validas:
            df = df.drop_duplicates(subset=colunas_validas)

        colunas_data = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col], format="%Y-%m-%d", errors='raise')
                colunas_data.append(col)
            except:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    colunas_data.append(col)
                except:
                    continue

        if colunas_data:
            st.sidebar.markdown("### 📅 Filtro por Data")
            coluna_data = st.sidebar.selectbox("Escolha a coluna de data", colunas_data)
            df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
            data_min = df[coluna_data].min()
            data_max = df[coluna_data].max()
            data_inicio, data_fim = st.sidebar.date_input("Intervalo de datas", [data_min, data_max], min_value=data_min, max_value=data_max)
            df = df[(df[coluna_data] >= pd.to_datetime(data_inicio)) & (df[coluna_data] <= pd.to_datetime(data_fim))]

        colunas_especie = [col for col in df.columns if "especie" in col.lower() or "espécie" in col.lower()]
        if colunas_especie:
            st.sidebar.markdown("### 🐾 Filtro por Espécie")
            coluna_especie = st.sidebar.selectbox("Escolha a coluna de espécie", colunas_especie)
            especies_unicas = sorted(df[coluna_especie].dropna().unique())
            especies_selecionadas = st.sidebar.multiselect("Seleciona uma ou mais espécies", especies_unicas)
            if especies_selecionadas:
                df = df[df[coluna_especie].isin(especies_selecionadas)]

        st.success("✅ Dados carregados com sucesso!")

        df.to_csv("dados_para_analise.csv", index=False)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Tabela", "📈 Estatísticas e Gráficos", "🔗 Juntar Tabelas", "🤖 Pedir Análise com GPT", "💻 Executar Código Python"])

        with tab1:
                    st.subheader("📄 Dados do Formulário")
                    st.dataframe(df, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### ➕ Carregar Dados Adicionais (CSV ou Excel)")

                    uploaded_file = st.file_uploader("Carregue um ficheiro .csv ou .xlsx", type=["csv", "xlsx"])

                    if uploaded_file is not None:
                        try:
                            if uploaded_file.name.endswith(".csv"):
                                df_extra = pd.read_csv(uploaded_file)
                            else:
                                df_extra = pd.read_excel(uploaded_file)

                            df_extra = normalizar_colunas(df_extra)

                            # Função para limpar nomes de colunas
                            import unicodedata
                            def normalizar_colunas(df):
                                df.columns = [unicodedata.normalize("NFKC", c).strip() for c in df.columns]
                                return df

                            df_extra = normalizar_colunas(df_extra)

                            st.success("✅ Dados adicionais carregados com sucesso!")
                            st.dataframe(df_extra, use_container_width=True)

                            # DEBUG: mostrar nomes reais das colunas
                            st.write("🧪 Colunas disponíveis no ficheiro externo:", df_extra.columns.tolist())

                            # Concatenar com dados do Kobo
                            # Guardar o ficheiro externo como base separada para merge
                            st.session_state.df_extra_raw = df_extra.copy()


                            colunas_validas = [col for col in df.columns if df[col].map(type).isin([str, int, float, bool, pd.Timestamp]).all()]
                            if colunas_validas:
                                df = df.drop_duplicates(subset=colunas_validas)

                            df.to_csv("dados_para_analise.csv", index=False)
                            st.info("📁 Os dados foram fundidos com os dados Kobo e atualizados para análise.")
                        except Exception as e:
                            st.error(f"Erro ao carregar o ficheiro: {e}")


        with tab2:

            st.subheader("📊 Estatísticas Interativas")
            colunas = df.columns.tolist()
            var = st.selectbox("Selecione a variável para analisar", colunas)

            if var:
                if pd.api.types.is_numeric_dtype(df[var]):
                    st.write("Estatísticas:")
                    st.write(df[var].describe())
                    st.bar_chart(df[var].value_counts())
                else:
                    st.write(df[var].value_counts())
                    st.bar_chart(df[var].value_counts())
            else:
                st.info("⚠️ Por favor, selecione uma variável para análise.")




        with tab3:
            st.subheader("🔗 Juntar dados externos aos dados Kobo")

            if 'df_extra_raw' not in st.session_state:
                st.warning("⚠️ Primeiro carregue um ficheiro externo na aba 📋 Tabela.")
            else:
                df_extra = st.session_state.df_extra_raw.copy()

                col_kobo = st.selectbox("🔹 Coluna no Kobo (df)", df.columns.tolist())
                col_extra = st.selectbox("🔸 Coluna nos Dados Externos (df_extra)", df_extra.columns.tolist())

                st.markdown("Seleciona as colunas do ficheiro externo que quer adicionar à base Kobo:")
                colunas_para_adicionar = st.multiselect(
                    "✅ Colunas a adicionar",
                    [c for c in df_extra.columns if c != col_extra]
                )

                if st.button("🚀 Executar Merge"):
                    try:
                        #st.write("📋 Colunas reais no df_extra:", df_extra.columns.tolist())
                        st.write("🔍 A fundir pela coluna (selecionada):", col_extra)

                        # Encontrar a coluna real por posição e não por igualdade direta
                        colunas_disponiveis = list(df_extra.columns)
                        col_extra_real = next(
                            (c for c in colunas_disponiveis if str(c).strip().lower() == str(col_extra).strip().lower()),
                            None
                        )

                        if col_extra_real is None:
                            st.error(f"❌ A coluna '{col_extra}' não foi encontrada no df_extra — mesmo com correspondência relaxada.")
                        else:
                            colunas_para_adicionar_reais = [
                                c for c in df_extra.columns
                                if c != col_extra_real and str(c).strip().lower() in [s.strip().lower() for s in colunas_para_adicionar]
                            ]

                            st.write("✅ Coluna real usada para o merge:", repr(col_extra_real))
                            #st.write("📥 Colunas a adicionar confirmadas:", colunas_para_adicionar_reais)

                            colunas_para_merge = [col_extra_real] + colunas_para_adicionar_reais

                            df_merge = pd.merge(
                                df,
                                df_extra.loc[:, colunas_para_merge],
                                left_on=col_kobo,
                                right_on=col_extra_real,
                                how='left'
                            )

                            df_merge = df_merge.drop(columns=[col_extra_real])
                            st.success("✅ Merge realizado com sucesso!")
                            st.dataframe(df_merge, use_container_width=True)

                            # Atualizar df e CSV
                            df = df_merge
                            df.to_csv("dados_para_analise.csv", index=False)
                            st.session_state.df = df.copy()


                    except Exception as e:
                        st.error(f"❌ Erro ao fazer merge: {e}")

        with tab4:
            st.subheader("🤖 Pedir Análise à OpenAI")
            df = st.session_state.get("df", pd.DataFrame())

            if not openai_api_key:
                st.warning("⚠️ Por favor, insira a chave da OpenAI na barra lateral.")
            else:
                st.markdown("<small>Seleciona as colunas a incluir na análise GPT (máximo recomendado: 5)</small>", unsafe_allow_html=True)
                colunas_gpt = st.multiselect("Colunas para incluir na análise GPT", df.columns.tolist(), max_selections=5)
                prompt = st.text_area("Escreva o que deseja perguntar sobre os dados selecionados")

                if st.button("🔍 Analisar com GPT") and prompt:
                    openai.api_key = openai_api_key

                    if not colunas_gpt:
                        st.warning("⚠️ Seleciona pelo menos uma coluna para enviar ao GPT.")
                    else:
                        df_gpt = df[colunas_gpt]
                        df_gpt.to_csv("dados_gpt_temp.csv", index=False)


                        colunas_str = ", ".join([f"'{col}'" for col in colunas_gpt])
                        pergunta_formatada = (
                            f"Analisa os dados do ficheiro 'dados_gpt_temp.csv'.\n\n"
                            f"As colunas selecionadas são: [{colunas_str}].\n"
                            f"Usa obrigatoriamente **estes nomes de colunas exatamente como estão**. "
                            f"Não inventes nomes novos nem alteres a grafia.\n\n"
                            f"Pergunta: {prompt}"
                        )

                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": pergunta_formatada
                        })

                        chat = openai.ChatCompletion.create(
                            model=modelo_openai,
                            temperature=0,
                            messages=st.session_state.chat_history,
                            max_tokens=1500
                        )

                        resposta = chat.choices[0].message.content
                        st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                        st.markdown(resposta)

            if len(st.session_state.chat_history) > 1:
                with st.expander("📜 Ver histórico da conversa"):
                    for msg in st.session_state.chat_history[1:]:
                        role = "👤 Utilizador:" if msg["role"] == "user" else "🤖 GPT:" 
                        st.markdown(f"**{role}** {msg['content']}")

        with tab5:
            st.subheader("💻 Executar Código Python")
            st.markdown("<small>Os dados do GPT estão guardados em 'dados_para_analise.csv'. Pode usar `df = pd.read_csv('dados_para_analise.csv')` diretamente.</small>", unsafe_allow_html=True)
            codigo_usuario = st.text_area("Cole aqui o código Python gerado ou escrito por si")
            if st.button("▶️ Executar Código") and codigo_usuario:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    try:
                        if "plt.show()" in codigo_usuario:
                            codigo_usuario = codigo_usuario.replace("plt.show()", "st.pyplot(fig)")
                            st.info("🔧 Substituímos automaticamente `plt.show()` por `st.pyplot(fig)`.")

                        if "st.pyplot()" in codigo_usuario and "st.pyplot(fig)" not in codigo_usuario:
                            codigo_usuario = codigo_usuario.replace("st.pyplot()", "st.pyplot(fig)")
                            st.info("🔧 Substituímos automaticamente `st.pyplot()` por `st.pyplot(fig)`.")

                        df = st.session_state.get("df", pd.DataFrame())
                        exec_globals = {"st": st, "plt": plt, "pd": pd, "df": df}

                        exec(codigo_usuario, exec_globals)

                        saida_execucao = buffer.getvalue()
                        if saida_execucao:
                            st.code(saida_execucao, language="text")

                        st.success("✅ Código executado com sucesso!")

                    except Exception as e:
                        st.error(f"❌ Erro ao executar o código: {e}")

    else:
        st.error(f"Erro ao carregar dados: {response.status_code}")
else:
    st.info("🔄 Insira o token e o UID do formulário para começar.")
