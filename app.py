import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configuração do tema do Streamlit
st.set_page_config(page_title="Análise de Satisfação do Cliente", layout="wide", initial_sidebar_state="expanded")

# Título e descrição da aplicação
st.title("🔍 Análise de Satisfação do Cliente")
st.markdown("Esta aplicação analisa a satisfação dos clientes de um restaurante, utilizando dados de um arquivo CSV. Você pode carregar seus dados, treinar um modelo de aprendizado de máquina e gerar insights valiosos para o seu negócio.")

# Carregar o arquivo CSV
uploaded_file = st.file_uploader("📁 Envie o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    dados = pd.read_csv(uploaded_file)

    # Exibir amostra do dataset
    st.subheader("📝 Amostra dos Dados")
    st.write(dados.head())

    # Verificar e preencher valores nulos, se houver
    if dados.isnull().sum().any():
        dados = dados.fillna(0)  # Preencher nulos com 0 (ou ajuste conforme necessário)
        st.write("🔴 Valores nulos detectados e preenchidos com 0.")

    # Gráficos iniciais
    st.subheader("📊 Visualização Inicial dos Dados")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Distribuição por gênero
        st.markdown("### Distribuição por Gênero")
        fig, ax = plt.subplots()
        sns.countplot(x='Gender', data=dados, ax=ax)
        st.pyplot(fig)

    with col2:
        # Distribuição de idade
        st.markdown("### Distribuição de Idade")
        fig, ax = plt.subplots()
        sns.histplot(dados['Age'], kde=True, bins=20, ax=ax)
        st.pyplot(fig)

    with col3:
        # Gasto médio por visita
        st.markdown("### Renda vs Gasto Médio")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Income', y='AverageSpend', data=dados, ax=ax)
        st.pyplot(fig)

    st.subheader("📈 Análise de Satisfação")
    
    col1, col2 = st.columns(2)

    with col1:
        # Análise de Satisfação com relação ao Tempo de Espera
        st.markdown("### Tempo de Espera vs Satisfação")
        fig, ax = plt.subplots()
        sns.boxplot(x='HighSatisfaction', y='WaitTime', data=dados, ax=ax)
        st.pyplot(fig)

    with col2:
        # Análise de Satisfação com relação à Qualidade do Serviço
        st.markdown("### Qualidade do Serviço vs Satisfação")
        fig, ax = plt.subplots()
        sns.boxplot(x='HighSatisfaction', y='ServiceRating', data=dados, ax=ax)
        st.pyplot(fig)

    st.subheader("📊 Fatores Impactantes na Satisfação")

    # Gráficos adicionais
    col1, col2, col3 = st.columns(3)

    with col1:
        # Satisfação por Gênero
        st.markdown("### Satisfação por Gênero")
        fig, ax = plt.subplots()
        sns.countplot(x='HighSatisfaction', hue='Gender', data=dados, ax=ax)
        ax.legend(title='Gênero')
        st.pyplot(fig)

    with col2:
        # Satisfação por Culinária Preferida
        st.markdown("### Satisfação por Culinária Preferida")
        fig, ax = plt.subplots()
        sns.countplot(y='PreferredCuisine', hue='HighSatisfaction', data=dados, ax=ax)
        ax.legend(title='Satisfação')
        st.pyplot(fig)

    with col3:
        # Satisfação por Tipo de Refeição
        st.markdown("### Satisfação por Tipo de Refeição")
        fig, ax = plt.subplots()
        sns.countplot(x='MealType', hue='HighSatisfaction', data=dados, ax=ax)
        ax.legend(title='Satisfação')
        st.pyplot(fig)

    if st.button("🚀 Treinar Modelos de Machine Learning e Gerar Insights"):
        # Remover espaços nos nomes das colunas
        dados.columns = dados.columns.str.strip()

        # Preparar os dados para machine learning
        X = dados.drop('HighSatisfaction', axis=1)
        y = dados['HighSatisfaction']
        X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding

        # Normalização dos dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir entre treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Modelos a serem testados
        modelos = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=200),
            'Support Vector Machine': SVC(),
            'Gradient Boosting': GradientBoostingClassifier()
        }

        resultados = {}

        # Treinamento e avaliação de modelos
        for nome, modelo in modelos.items():
            pipeline = Pipeline([('classifier', modelo)])
            param_grid = {}
            
            # Definindo parâmetros para cada modelo
            if nome == 'Random Forest':
                param_grid = {'classifier__n_estimators': [50, 100]}
            elif nome == 'Logistic Regression':
                param_grid = {'classifier__C': [0.1, 1, 10]}
            elif nome == 'Support Vector Machine':
                param_grid = {'classifier__kernel': ['linear', 'rbf'], 'classifier__C': [0.1, 1, 10]}
            elif nome == 'Gradient Boosting':
                param_grid = {'classifier__learning_rate': [0.01, 0.1, 1], 'classifier__n_estimators': [50, 100]}

            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Previsões e métricas
            y_pred = grid_search.predict(X_test)
            resultados[nome] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }

        # Apresentar resultados
        st.subheader("📊 Comparação de Modelos")
        for nome, resultado in resultados.items():
            st.markdown(f"### {nome}")
            st.write(f"**Acurácia:** {resultado['accuracy']:.2f}")
            st.write("**Relatório de Classificação:**")
            st.text(classification_report(y_test, y_pred, target_names=["Baixa", "Alta"]))
        
        # Matriz de confusão para o melhor modelo
        melhor_modelo = max(resultados.items(), key=lambda x: x[1]['accuracy'])[0]
        y_pred_melhor = grid_search.predict(X_test)  # Corrigido para pegar as previsões do melhor modelo

        st.subheader("🔍 Matriz de Confusão")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_melhor), annot=True, cmap='Blues', ax=ax)
        ax.set_title("Matriz de Confusão do Melhor Modelo")
        st.pyplot(fig)

        # Importância das variáveis para o Random Forest
        if 'Random Forest' in resultados:
            modelo_rf = modelos['Random Forest']
            modelo_rf.fit(X_train, y_train)
            importancias = modelo_rf.feature_importances_
            indices = pd.Series(importancias, index=X.columns).sort_values(ascending=False)

            st.subheader("🌟 Importância das Variáveis")
            fig, ax = plt.subplots()
            sns.barplot(x=indices, y=indices.index, palette='viridis', ax=ax)
            ax.set_title("Fatores que mais Impactam a Satisfação")
            st.pyplot(fig)

            # Geração de Insights
            st.subheader("💡 Insights para Melhorar a Satisfação dos Clientes")
            st.write("1. **Treinamento de Equipe:** Aumentar a frequência de treinamentos pode melhorar a qualidade do atendimento.")
            st.write("2. **Monitorar o Tempo de Espera:** Implementar um sistema para otimizar o tempo de espera dos clientes.")
            st.write("3. **Personalização de Ofertas:** Utilizar dados de visitas e preferências para criar promoções personalizadas.")
            st.write("4. **Melhorar a Qualidade da Comida:** Investir na qualidade dos pratos oferecidos com base nas preferências dos clientes.")
            st.write("5. **Feedback Contínuo:** Criar um sistema para coletar feedback após cada visita para ajustes contínuos.")
