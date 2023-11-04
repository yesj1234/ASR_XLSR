import argparse
import torch
import librosa
from tqdm import tqdm
import logging 
import sys
import re 

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer
)
from datasets import load_dataset 
import evaluate

logger = logging.getLogger(__name__)
 # Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


CHARS_TO_IGNORE_REGEX = re.compile("[+-。、「」【】〜〉…？?.,~!]")

def remove_special_characters(batch):
    batch["target_text"] = re.sub(CHARS_TO_IGNORE_REGEX, "", batch["target_text"])
    return batch

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["file"], sr=16_000)
    batch["audio"] = speech_array
    batch["target_text"] = batch["target_text"]
    return batch
    
def main():
    raw_dataset = load_dataset("./sample_speech.py", split="test")
    raw_dataset = raw_dataset.map(remove_special_characters, num_proc = 8, desc="remove special chars")
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
        
        
    predictions = list(map(lambda x: re.sub(CHARS_TO_IGNORE_REGEX, "", x), predictions))
    
    
    with open("predictions.txt", "w+", encoding="utf-8") as f:
        for prediction, reference in zip(predictions, references):
            f.write(f"{prediction} :: {reference}\n")
    
    for i, pair in enumerate(zip(predictions, references)):
        prediction, reference =pair
        prediction = prediction.strip()
        prediction = " ".join(list(prediction))
        reference = reference.strip()
        reference = " ".join(list(reference))         
        if len(prediction) < 1 or len(reference) < 1:
            prediction.pop(i)
            reference.pop(i)
        else:
            logger.warning(f"Prediction or reference is empty")
            logger.warning(f"prediction: {prediction}")
            logger.warning(f"reference : {reference}")
    
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")
    try:
        cer_score = cer.compute(predictions = predictions, references = references)
        wer_score = wer.compute(predictions = predictions, references = references)
        logger.info(f"cer: {cer_score}")
        logger.info(f"wer: {wer_score}")
    except Exception as e:
        print(e)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine tuned model dir. relative dir path, or repo_id from huggingface")
    args = parser.parse_args()
    main()
