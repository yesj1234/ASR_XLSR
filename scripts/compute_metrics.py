import argparse
import torch
import librosa
from tqdm import tqdm

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer
)
from datasets import load_dataset 
import evaluate

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["file"], sr=16_000)
    batch["audio"] = speech_array
    batch["target_text"] = batch["target_text"]
    return batch


    
def main():
    raw_dataset = load_dataset("./test_loader.py", split="test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)
    processor = Wav2Vec2Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
    
    references = raw_dataset["target_text"]
    predictions = []
    
    vectorized_dataset = raw_dataset.map(
        speech_file_to_array_fn,
        num_proc=8,
        desc="preprocess datasets"
    )
    
    def generate_predictions(batch):
        inputs = processor(batch["audio"], sampling_rate=16_000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask = inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim = -1)
        predicted_sentences = processor.batch_decode(predicted_ids)
        return predicted_sentences
    
    for batch in tqdm(vectorized_dataset):
        predicted_sentences = generate_predictions(batch)
        predictions.append(*predicted_sentences)
    
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")
    cer_score = cer.compute(predictions = predictions, references = references)
    wer_score = wer.compute(predictions = predictions, references = references)
    print(f"cer: {cer_score}")
    print(f"wer: {wer_score}")
    # cer_score = cer.compute(predictions = predictions, references = )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine tuned model dir. relative dir path, or repo_id from huggingface")
    args = parser.parse_args()
    
    
    
    main()
