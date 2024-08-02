import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from jiwer import wer

def transcrever_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="pt-BR")
    except sr.UnknownValueError:
        print("Não foi possível entender o áudio")
    except sr.RequestError:
        print("Erro ao conectar-se ao serviço de reconhecimento de fala")
    return None

def gerar_texto_gpt2(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calcular_wer(referencia, transcricao):
    return wer(referencia, transcricao)

if __name__ == "__main__":
    # Caminho para o arquivo de áudio
    audio_path = "/content/"

    # Transcrição do áudio
    transcricao = transcrever_audio(audio_path)
    print(f"Transcrição: {transcricao}")

    if transcricao:
        # Geração de texto com GPT-2 usando a transcrição como prompt
        resposta_gpt2 = gerar_texto_gpt2(transcricao)
        print(f"Resposta do GPT-2: {resposta_gpt2}")

        # Cálculo do Word Error Rate (WER)
        wer_valor = calcular_wer(transcricao, resposta_gpt2)
        print(f"Word Error Rate (WER): {wer_valor}")
