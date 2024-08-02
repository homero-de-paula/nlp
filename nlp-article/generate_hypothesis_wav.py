import argparse
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from gpt2_inference import GPT2Inference

def transcribe_audio(audio_path, processor, model):
    speech, rate = sf.read(audio_path)
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main(args):
    # Load Wave2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)

    # Transcribe audio
    transcription = transcribe_audio(args.data_folder, processor, model)
    print(f"Transcription: {transcription}")

    # Load GPT-2 model for further text generation
    gpt2 = GPT2Inference(model_name='gpt2')
    generated_text = gpt2.generate_text(transcription)
    print(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, required=True, help='Type of data')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    args = parser.parse_args()
    main(args)
