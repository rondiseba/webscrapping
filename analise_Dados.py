import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import font_manager

# Carregar uma fonte que suporte emojis
font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
font_manager.fontManager.ttflist  # Lista de fontes
plt.rcParams["font.family"] = "DejaVu Sans"  # Fonte padrão no matplotlib

# Carregar os dados tratados
df = pd.read_csv("vagas_remote_ok_tratadas.csv")

# Visualização de Dados
def visualizar_dados(df):
    # Configuração para visualizações
    sns.set_theme(style="whitegrid")

    # 1. Distribuição de Vagas por Localidade
    plt.figure(figsize=(10, 6))
    df["Local"].value_counts().head(10).plot(kind="bar", color="teal")
    plt.title("Distribuição de Vagas por Localidade (Top 10)")
    plt.ylabel("Quantidade")
    plt.xlabel("Localidade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Faixa Salarial (histograma)
    df_salarios = df[df["Salário"].str.contains(r"\d")]  # Apenas linhas com números
    df_salarios["Salário"] = df_salarios["Salário"].str.extract(r"(\d+)").astype(float)  # Extrai o valor mínimo
    plt.figure(figsize=(10, 6))
    sns.histplot(df_salarios["Salário"], bins=20, kde=True, color="purple")
    plt.title("Distribuição de Salários")
    plt.xlabel("Salário")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

    # 3. Empresas com Mais Vagas
    plt.figure(figsize=(10, 6))
    df["Empresa"].value_counts().head(10).plot(kind="bar", color="orange")
    plt.title("Empresas com Mais Vagas (Top 10)")
    plt.ylabel("Quantidade")
    plt.xlabel("Empresa")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Modelagem de Dados
def modelar_dados(df):
    # Preparação dos dados para classificação
    df["Categoria"] = df["Título"].str.contains("Data|Analyst|Python", case=False).astype(int)  # Exemplo de categorização
    X = pd.get_dummies(df[["Local", "Salário"]], drop_first=True)  # Transformação de dados categóricos
    y = df["Categoria"]

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classificação (Árvore de Decisão)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia da Classificação: {acc:.2f}")

    # Regressão para Estimativa Salarial
    if "Salário" in X.columns:
        df_salarios = df[df["Salário"].str.contains(r"\d")]  # Apenas linhas com salários numéricos
        df_salarios["Salário"] = df_salarios["Salário"].str.extract(r"(\d+)").astype(float)  # Extrai o valor mínimo
        X_reg = pd.get_dummies(df_salarios[["Local"]], drop_first=True)  # Apenas local para regressão
        y_reg = df_salarios["Salário"]

        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        reg = LinearRegression()
        reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = reg.predict(X_test_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        print(f"Erro Quadrático Médio na Regressão Salarial: {mse:.2f}")

# Executar Análise
if __name__ == "__main__":
    visualizar_dados(df)
    modelar_dados(df)
