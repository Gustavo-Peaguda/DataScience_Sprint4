# TOTVS AdvisorMind — Data Science Challenge 2026

## Sobre o projeto

O **TOTVS AdvisorMind** é um projeto acadêmico desenvolvido para o Challenge 2026 da FIAP, na disciplina de **Data Science and Statistical Computing**, do curso de Engenharia de Software.

O projeto utiliza técnicas de **Processamento de Linguagem Natural (NLP)**, análise de dados e **Machine Learning** para transformar transcrições de reuniões comerciais em informações úteis para a tomada de decisões.

A solução permite analisar:

- **Produtos TOTVS identificados** nas transcrições;
- **Sentimento** presente nas conversas;
- **Risco de insatisfação**, utilizado como indicador experimental relacionado a churn;
- **Oportunidades comerciais**;
- **Principais termos** encontrados;
- **Recomendações de ações comerciais**.

O desenvolvimento foi organizado em quatro sprints, contemplando a preparação dos dados, análise exploratória, comparação de algoritmos supervisionados e integração da solução em um painel interativo desenvolvido com **Streamlit**.

> **Observação:** o sistema é um protótipo acadêmico. O modelo de previsão de risco possui limitações de desempenho e pode apresentar resultados inconclusivos quando a transcrição não contém informações suficientes.

## Como executar o projeto

### 1. Pré-requisitos

É necessário ter instalado:

- Python 3.11 ou superior, preferencialmente uma versão compatível com o `requirements.txt`;
- Git;
- Visual Studio Code (recomendado).

### 2. Clonar o repositório

Abra o terminal do VS Code e execute:

```bash
git clone https://github.com/Gustavo-Peaguda/DataScience_Sprint4.git
cd DataScience_Sprint4
```

### 3. Criar o ambiente virtual

Execute o comando:

```bash
python -m venv .venv
```

O ambiente virtual permite instalar as dependências do projeto de forma isolada.

### 4. Instalar as dependências

No terminal do VS Code, utilizando Windows e PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Esse comando instala as bibliotecas necessárias diretamente no ambiente virtual, sem exigir sua ativação manual.

### 5. Executar o Streamlit

Após a instalação, execute:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Quando o servidor iniciar, abra o endereço:

**http://localhost:8501**

O painel interativo permitirá consultar dados, visualizar os resultados dos modelos e realizar análises de novas transcrições comerciais.

### 6. Executar os notebooks (opcional)

Para visualizar os estudos e as análises desenvolvidas nas sprints, abra os arquivos `.ipynb` no VS Code utilizando as extensões Python e Jupyter.

Selecione o interpretador Python do ambiente `.venv` e execute as células na ordem apresentada.

## Tecnologias utilizadas

- **Python** — desenvolvimento do projeto;
- **Pandas e NumPy** — processamento e manipulação dos dados;
- **Scikit-learn** — treinamento e avaliação dos modelos;
- **TF-IDF** — representação textual;
- **Matplotlib e Seaborn** — gráficos e visualizações;
- **Streamlit** — interface interativa;
- **Jupyter Notebook** — experimentação e documentação.

## Integrantes

| Nome | RM |
|---|---|
| Enzo Fernandes Ramos | 563705 |
| Felipe Henrique de Souza Cerazi | 562746 |
| Gustavo Peaguda de Castro | 562923 |
| Lorenzo Coque | 563385 |

## Informações acadêmicas

**Instituição:** FIAP — Faculdade de Informática e Administração Paulista  
**Curso:** Engenharia de Software  
**Disciplina:** Data Science and Statistical Computing  
**Projeto:** TOTVS AdvisorMind  
**Challenge:** TOTVS 2026  
**Ano:** 2026
