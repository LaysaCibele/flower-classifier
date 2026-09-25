# 🌷 Flower Classifier

Sistema completo de classificação de flores que integra **Machine Learning, desenvolvimento web, banco de dados e agentes de IA** em uma única aplicação.

## Sobre o projeto

O **Flower Classifier** é um projeto de estudo e portfólio desenvolvido para integrar conhecimentos de **Ciência da Computação, Ciência de Dados, Machine Learning e Desenvolvimento de Software** em uma aplicação completa.

O sistema permitirá que o usuário informe características de uma flor, obtenha sua classificação por meio de um modelo de Machine Learning, consulte o histórico de classificações e interaja com um agente de IA capaz de explicar as previsões e responder perguntas sobre a classificação.

O projeto será desenvolvido de forma incremental, passando pelo desenvolvimento do modelo de Machine Learning, frontend, backend, banco de dados, agente de IA, containerização e deploy.

## Funcionalidades

*  Classificação de espécies de flores utilizando Machine Learning
*  Interface web desenvolvida com React
*  API REST para comunicação entre as aplicações
*  Backend em Python
*  Banco de dados para armazenamento do histórico de classificações
*  Agente de IA para explicar previsões e responder perguntas
*  Base de conhecimento utilizando RAG
*  Ferramentas integradas ao agente de IA
*  Containerização com Docker
*  Deploy em ambiente de nuvem
*  Testes da API utilizando Postman

## Arquitetura

```text
                         ┌─────────────────┐
                         │      React      │
                         │    Frontend     │
                         └────────┬────────┘
                                  │
                             REST API
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     Backend     │
                         │      API        │
                         └───────┬─────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │ ML Model    │  │  Database   │  │ AI Agent    │
        │ Classifier  │  │             │  │             │
        └─────────────┘  └─────────────┘  └──────┬──────┘
                                                  │
                                                  ▼
                                           ┌─────────────┐
                                           │ Base de     │
                                           │ conhecimento│
                                           │ / RAG       │
                                           └─────────────┘
```

## Machine Learning

O projeto utiliza o **Iris Dataset** para desenvolver um modelo de classificação multiclasse.

O modelo recebe quatro características da flor:

* Comprimento da sépala
* Largura da sépala
* Comprimento da pétala
* Largura da pétala

E realiza a previsão de uma das três espécies:

* Iris setosa
* Iris versicolor
* Iris virginica

O pipeline de Machine Learning incluirá:

1. Carregamento dos dados
2. Análise exploratória dos dados
3. Preparação dos dados
4. Separação entre treino e teste
5. Treinamento do modelo
6. Avaliação
7. Serialização do modelo
8. Integração com a API

## Agente de IA

O agente de IA será responsável por explicar as previsões realizadas pelo modelo e responder perguntas relacionadas à classificação das flores.

Em vez de treinar um modelo de linguagem do zero, o projeto utilizará uma abordagem de **RAG (Retrieval-Augmented Generation)**.

Será criada uma base de conhecimento contendo informações sobre:

* Espécies de Iris
* Características das flores
* Diferenças entre as espécies
* Interpretação das medidas
* Conceitos de Machine Learning utilizados no projeto

Durante uma interação, o agente poderá buscar informações relevantes nessa base e utilizá-las junto aos dados fornecidos pela própria aplicação para construir sua resposta.

Também será explorado o uso de **tool calling**, permitindo que o agente utilize funções da aplicação, como consultar o histórico de classificações ou acessar informações de uma determinada previsão.

## Tecnologias

### Frontend

* React
* JavaScript
* HTML
* CSS

### Backend

* Python
* FastAPI
* REST API

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

### Inteligência Artificial

* Large Language Models (LLMs)
* RAG
* Embeddings
* Busca vetorial
* Tool calling / Function calling

### Banco de Dados

* SQL
* SQLite durante o desenvolvimento
* PostgreSQL como opção para produção

### DevOps

* Git
* GitHub
* Docker
* Docker Compose

### Testes e integração

* REST
* Postman

### Deploy

* Vercel para o frontend
* Serviço de nuvem para o backend e demais componentes

## Estrutura do projeto

```text
flower-classifier/
│
├── frontend/
│   └── Aplicação React
│
├── backend/
│   └── API REST
│
├── ml/
│   ├── Treinamento do modelo
│   ├── Análise dos dados
│   └── Artefatos de Machine Learning
│
├── README.md
├── LICENSE
└── .gitignore
```

A estrutura poderá ser modificada conforme novas funcionalidades forem adicionadas ao projeto.

## Objetivos de aprendizado

Além de funcionar como um projeto de portfólio, o Flower Classifier será utilizado como um ambiente prático para consolidar conhecimentos de diferentes áreas da Computação.

A proposta é conectar:

```text
Programação
      ↓
Algoritmos e Estruturas de Dados
      ↓
Machine Learning
      ↓
Backend e APIs
      ↓
Banco de Dados
      ↓
Frontend
      ↓
Arquitetura de Software
      ↓
Agentes de IA
      ↓
Docker e Cloud
```

O projeto será desenvolvido de forma gradual, buscando compreender cada parte da aplicação em vez de utilizar uma solução pronta para todo o sistema.

## Licença

Este projeto está sob a licença MIT.
