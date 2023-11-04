import argparse
import torch
import librosa
from tqdm import tqdm
import logging 
import sys
import re 
from time import time 

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
logger.setLevel(logging.INFO)


CHARS_TO_IGNORE_REGEX = {
    "ko": re.compile("[.?!.,]"),
    "ja": re.compile("[+-。、「」【】〜〉…？?.,~!]"),
    "zh": re.compile("[。？，！.,?~]"),
    "en": re.compile("[.,?!~]")
}


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["file"], sr=16_000)
    batch["audio"] = speech_array
    batch["target_text"] = batch["target_text"]
    return batch
    
def main(args):
    start_time = time()
    special_chars = CHARS_TO_IGNORE_REGEX[args.lang]
    def remove_special_characters(batch):
        batch["target_text"] = re.sub(special_chars, "", batch["target_text"])
        return batch

    raw_dataset = load_dataset("./sample_speech.py", split="validation")
    raw_dataset = raw_dataset.map(remove_special_characters, num_proc = 8, desc="remove special chars")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)
    processor = Wav2Vec2Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
    
    references_temp = []
    predictions_temp = []
    
    
    vectorized_dataset = raw_dataset.map(
        speech_file_to_array_fn,
        num_proc=8,
        desc="preprocess datasets"
    )
    
    def generate_predictions(batch):
        inputs = processor(batch["audio"], sampling_rate=16_000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask = inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, axis = -1)
        predicted_sentences = processor.batch_decode(predicted_ids)
        return predicted_sentences
    
    empty_files = []
    
    logger.info("***** Running Evaluation *****")
    for batch in tqdm(vectorized_dataset):
        try:
            predicted_sentence = generate_predictions(batch)
            predicted_sentence = predicted_sentence[0].strip()
            if len(predicted_sentence) < 1:
                empty_files.append(batch["file"])
            else:
                predictions_temp.append(predicted_sentence)
                references_temp.append(batch["target_text"])
        except Exception as e:
            logger.warning(e)
            pass
        
    predictions_temp = list(map(lambda x: re.sub(special_chars, "", x), predictions_temp))
    
    with open("empty_files.txt", "w+", encoding="utf-8") as f:
        for path in empty_files:
            f.write(f"{path}\n")
    
    with open("predictions.txt", "w+", encoding="utf-8") as f:
        for prediction, reference in zip(predictions_temp, references_temp):
            f.write(f"{prediction} :: {reference}\n")
            
    predictions, references = [], []
    
    logger.info("***** Simple postprocessing *****")
    for i, pair in tqdm(enumerate(zip(predictions_temp, references_temp))):
        prediction, reference =pair
        prediction = prediction.strip()
        prediction = " ".join(list(prediction))
        reference = reference.strip()
        reference = " ".join(list(reference))         
        if len(prediction) >0 and len(reference) > 0:
            predictions.append(prediction)
            references.append(reference)
        else:
            logger.warning(f"Prediction or reference is empty")
    
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")
    try:
        cer_score = cer.compute(predictions = predictions, references = references)
        wer_score = wer.compute(predictions = predictions, references = references)
        logger.info(f"""
                    ***** eval metrics *****
                      eval_samples: {len(predictions)}
                      eval_cer    : {cer_score}
                      eval_wer    : {wer_score}
                      eval_runtime: {time() - start_time} 
                    """)
    except Exception as e:
        print(e)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine tuned model dir. relative dir path, or repo_id from huggingface")
    parser.add_argument("--lang", help="ko ja zh en")
    args = parser.parse_args()
    main(args)
