# Language Model Evaluation

## Descrição
Este repositório armazena um projeto que utiliza GPT-2 como modelo de linguagem para o Wave2Vec para a língua portuguesa.

## Dados Utilizados
- **Wikipedia Dump**: [Download](http://www02.smt.ufrj.br/~igor.quintanilha/ptwiki-20181125.txt)
- **CETUC**: [Download](http://www02.smt.ufrj.br/~igor.quintanilha/alcaim.tar.gz)
- **Common Voice**: [Download](https://commonvoice.mozilla.org/pt/datasets)
- **CORAA**: [Download](https://github.com/nilc-nlp/CORAA)
- **MLS**: [Download](http://www.openslr.org/94/)

## Instalação de Dependências
Primeiro, instale as dependências necessárias:

pip install -r requirements.txt
pip install -r language-model-evaluation/requirements.txt

## Coraa 

python3 generate_hypothesis.py \
    --data_type commonvoice \
    --data_folder ./language-model-evaluation/data/cv-corpus-6.1-2020-12-11/pt \
    --model_name ./language-model-evaluation/models/wave2vec_model \
    --output_path ./language-model-evaluation/hypothesis/cv-6.1-w2v-cv-6.1-coraa \ 
