import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Configuração de cabeçalhos
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# URLs para as buscas
urls = [
    "https://remoteok.io/remote-data-jobs",  # Vagas de análise e ciência de dados
    "https://remoteok.io/remote-python-jobs",  # Vagas relacionadas a Python
    "https://remoteok.io/remote-backend-jobs"  # Vagas de backend
]

def obter_vagas_remote_ok(url):
    try:
        # Realiza a requisição HTTP
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Coleta os dados das vagas
        vagas = []
        for job in soup.find_all("tr", class_="job"):  # Cada vaga está em uma tag <tr> com classe 'job'
            titulo = job.find("h2", itemprop="title").get_text(strip=True)
            empresa = job.find("h3", itemprop="name").get_text(strip=True)
            local = job.find("div", class_="location").get_text(strip=True) if job.find("div", class_="location") else "Remoto"
            salario = job.find("div", class_="salary").get_text(strip=True) if job.find("div", class_="salary") else "Não informado"
            link = "https://remoteok.io" + job.find("a", class_="preventLink")["href"]

            vagas.append({
                "Título": titulo,
                "Empresa": empresa,
                "Local": local,
                "Salário": salario,
                "Link": link
            })
        
        return vagas

    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar {url}: {e}")
        return []

# Limitar requisições e combinar dados
def capturar_dados():
    print("Iniciando coleta de vagas...")
    todas_vagas = []
    for url in urls:
        vagas = obter_vagas_remote_ok(url)
        todas_vagas.extend(vagas)
        print(f"Coletado {len(vagas)} vagas de {url}.")
        time.sleep(6)  # Respeitando o limite de requisições
    
    if todas_vagas:
        print(f"Total de vagas coletadas: {len(todas_vagas)}")
        return pd.DataFrame(todas_vagas)
    else:
        print("Nenhuma vaga encontrada.")
        return pd.DataFrame()

# Tratamento de dados
def tratar_dados(df):
    if df.empty:
        print("Nenhum dado para tratar.")
        return df

    # Remoção de duplicatas
    df = df.drop_duplicates(subset=["Título", "Link"])
    
    # Tratamento de valores faltantes
    df["Local"].fillna("Não informado", inplace=True)
    df["Salário"].fillna("Não informado", inplace=True)

    # Normalização de strings
    df["Título"] = df["Título"].str.strip().str.title()
    df["Empresa"] = df["Empresa"].str.strip().str.title()
    df["Local"] = df["Local"].str.strip().str.title()
    df["Salário"] = df["Salário"].str.strip()

    # Transformação de salários para um formato padrão
    df["Salário"] = df["Salário"].str.replace(r"[^\d\-]", "", regex=True)  # Remove caracteres não numéricos

    return df

# Executar o script
if __name__ == "__main__":
    df_vagas = capturar_dados()
    if not df_vagas.empty:
        print("Dados antes do tratamento:")
        print(df_vagas.head())
        
        # Aplicar tratamento de dados
        df_vagas_tratados = tratar_dados(df_vagas)
        
        print("Dados após o tratamento:")
        print(df_vagas_tratados.head())
        
        # Salva o DataFrame em um arquivo CSV
        df_vagas_tratados.to_csv("vagas_remote_ok_tratadas.csv", index=False, encoding="utf-8")
        print("Arquivo salvo como 'vagas_remote_ok_tratadas.csv'.")
    else:
        print("Nenhum dado foi salvo.")
