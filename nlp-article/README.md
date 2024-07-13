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

```sh
pip install -r requirements.txt

#Realizar importação dos modelos 

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
