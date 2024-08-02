from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from jiwer import wer

def carregar_dados(path_data):
    # Carrega os dados do arquivo de texto
    with open(path_data, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

def preparar_dataset(path_data, tokenizer, block_size=128):
    # Prepara o dataset para o treinamento
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=path_data,
        block_size=block_size
    )
    return dataset

def treinar_modelo(tokenizer, dataset, output_dir="gpt2_model"):
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    return model

def gerar_texto(model, tokenizer, prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calcular_wer(referencia, transcricao):
    return wer(referencia, transcricao)

if __name__ == "__main__":
    path_data = "caminho_para_os_dados.txt"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = preparar_dataset(path_data, tokenizer)
    modelo_treinado = treinar_modelo(tokenizer, dataset)    
    dados_validacao = carregar_dados(path_data)
    
    wer_total = 0
    n_amostras = len(dados_validacao)
    
    for referencia in dados_validacao:
        transcricao = gerar_texto(modelo_treinado, tokenizer, referencia)
        wer_valor = calcular_wer(referencia, transcricao)
        wer_total += wer_valor
        print(f"Referência: {referencia}")
        print(f"Transcrição: {transcricao}")
        print(f"WER: {wer_valor}\n")
    
    wer_medio = wer_total / n_amostras
    print(f"WER Médio: {wer_medio}")
