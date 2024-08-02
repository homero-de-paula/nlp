# Language Model Evaluation

## Descrição
Este repositório armazena um projeto que utiliza GPT-2 em comparação ao Wave2Vec para a língua portuguesa.

## Dados Utilizados
- **CETUC**: [Download](http://www02.smt.ufrj.br/~igor.quintanilha/alcaim.tar.gz)
- **CORAA**: [Download](https://github.com/nilc-nlp/CORAA)
- **COMMON VOICE**: [Download](https://commonvoice.mozilla.org/pt/datasets)
## Instalação de Dependências
Primeiro, instale as dependências necessárias:

pip install -r requirements.txt

## Usar individual

python3 gpt2.py > Dentro do arquivo substitua o diretorio de onde está o audio_file  
     
    
## Usar datasets combinados

Agora você pode usar combine_datasets.py para gerar combinações de todos os conjuntos de dados e estimar variações do gpt2.
Existe também a possibilidade de utilizar o gpt-2 apenas em raw data com raw.py
