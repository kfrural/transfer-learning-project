# Transfer Learning Project

Este projeto demonstra como aplicar **Transfer Learning** para classificar imagens usando um modelo pré-treinado, especificamente o **MobileNetV2**, com o objetivo de classificar categorias de imagens em um conjunto de dados fornecido.

## Estrutura do Projeto

A estrutura do projeto é organizada da seguinte maneira:

- `data/`: Contém os dados de treinamento e validação.
    - `train/`: Diretório com imagens de treinamento, divididas por categoria.
    - `validation/`: Diretório com imagens de validação, divididas por categoria.
- `src/`: Contém os scripts principais de código.
    - `model.py`: Definição do modelo de Transfer Learning.
    - `data_utils.py`: Funções para carregar e processar os dados.
    - `main.py`: Script principal para treinar o modelo e visualizar os resultados.
- `utils/`: Funções auxiliares para download de dados e plotagem de resultados.
    - `download_data.py`: Script para baixar e extrair o conjunto de dados.
    - `plot_results.py`: Funções para visualizar os gráficos de desempenho durante o treinamento.
- `requirements.txt`: Lista de dependências necessárias para rodar o projeto.
- `.gitignore`: Arquivos e diretórios que não serão versionados, como dados e caches.
- `README.md`: Documentação do projeto.

## Pré-Requisitos

Certifique-se de ter os seguintes pré-requisitos instalados no seu sistema:

- Python 3.x
- Pip (para gerenciar pacotes)

## Instalação

1. **Clone este repositório:**

```bash
git clone https://github.com/seu-usuario/transfer-learning-project.git
cd transfer-learning-project
```

2. **Crie e ative um ambiente virtual:**

```bash
python -m venv venv
source venv/bin/activate  # no Windows use venv\Scripts\activate
```

3. **Instale as dependências do projeto:**

```bash
pip install -r requirements.txt
```

## Baixar e Preparar os Dados

1. **Baixe e extraia os dados:**

O script `download_data.py` pode ser utilizado para baixar e extrair os dados diretamente de uma URL fornecida.

- Modifique o link da variável `url` no arquivo `download_data.py` para a URL do seu conjunto de dados.
- Depois, execute o script para baixar os dados:

```bash
python utils/download_data.py
```

2. **Organize os dados:**

Certifique-se de que as imagens estejam organizadas em diretórios de treinamento (`train/`) e validação (`validation/`), conforme a estrutura esperada.

## Treinamento do Modelo

1. **Execute o script principal para treinar o modelo:**

```bash
python src/main.py
```

Isso carregará os dados de treinamento e validação, treinará o modelo usando Transfer Learning com MobileNetV2 e exibirá gráficos de desempenho (loss e accuracy) durante o treinamento.

## Visualização de Resultados

Durante o treinamento, gráficos serão gerados automaticamente para mostrar a evolução do **Loss** e da **Accuracy** para o conjunto de treinamento e validação.

## Estrutura do Modelo

O modelo usa o **MobileNetV2** como base, removendo a camada final (topo) e adicionando novas camadas para adaptação ao seu problema específico. O modelo é compilado usando o otimizador **Adam** e a função de perda **categorical_crossentropy**.

## Contribuições

Sinta-se à vontade para contribuir com o projeto! Você pode:
- Abrir issues para sugerir melhorias ou relatar bugs.
- Fazer um fork e enviar um pull request com suas contribuições.

