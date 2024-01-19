import argparse
import torch
import librosa
from tqdm import tqdm
import logging 
import sys
import re 
import os
from time import time 
import traceback

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

METRIC_MAPPER = {
    "ko": "cer",
    "ja": "cer",
    "zh": "cer",
    "en": "wer"
}

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["file"], sr=16_000)
    batch["audio"] = speech_array
    batch["target_text"] = batch["target_text"]
    return batch

def main(args):
    start_time = time()
    special_chars = CHARS_TO_IGNORE_REGEX[args.lang]
    
    # 1. load the dataset
    raw_dataset = load_dataset(args.load_script, trust_remote_code=True)
    current_split = list(raw_dataset.data.keys())[0]
    raw_dataset = raw_dataset[current_split].filter(lambda x: x["duration"] >= 2, 
                                                    desc = "filter wav file less than 2 seconds.") # filter out wav files that are less than 2 seconds. 
    # 2. remove special characters that doesn't have any phoneme.
    def remove_special_characters(batch):
        batch["target_text"] = re.sub(special_chars, "", batch["target_text"])
        return batch
    raw_dataset = raw_dataset.map(remove_special_characters, num_proc = 8, desc="remove special chars")

    # 3. load the model, load the feature extractor and tokenizer and put in processor for simple use. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)
    processor = Wav2Vec2Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
        
    # 4. vectorize the loaded raw dataset.
    vectorized_dataset = raw_dataset.map(
        speech_file_to_array_fn,
        num_proc=8,
        desc="preprocess datasets"
    )

    # 5. generate predictions in batch.
    def generate_predictions(batch):
        with torch.device("cuda"): 
            inputs = processor(batch["audio"], sampling_rate=16_000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask = inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, axis = -1)
        predicted_sentence = processor.batch_decode(predicted_ids)
        batch["predicted_sentence"] = predicted_sentence
        return batch
    
    predicted_datasets = vectorized_dataset.map(generate_predictions, 
                           batched=True, 
                           batch_size = 10,
                           desc="running prediction")
    def post_processing(batch):
        batch["predicted_sentence"] = list(map(lambda x: " ".join(x.split()), batch["predicted_sentence"])) # remove unnecessary white spaces between words if exists. 
        batch["predicted_sentence"] = list(map(lambda x: x.lower(), batch["predicted_sentence"])) 
        batch["predicted_sentence"] = list(map(lambda x: re.sub(CHARS_TO_IGNORE_REGEX[args.lang], "", x), batch["predicted_sentence"])) # remove special chars

        batch["target_text"] = list(map(lambda x: " ".join(x.split()), batch["target_text"]))
        batch["target_text"] = list(map(lambda x: x.lower(), batch["target_text"]))
        return batch
    
    predicted_datasets = predicted_datasets.map(post_processing, batched = True, batch_size = 1000, desc="simple post processing")
    
    predictions = predicted_datasets["predicted_sentence"]
    references = predicted_datasets["target_text"]
    paths = list(map(lambda x: x["path"], raw_dataset['audio']))

    
    cur_metric = METRIC_MAPPER[args.lang]
    metric = evaluate.load(cur_metric)
    
    empty_indexes = []
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        if len(prediction.strip()) == 0 or len(reference.strip()) == 0: # filter any possible empty predictions due to some data issues.
          empty_indexes.append(i)  
            
    predictions = [pred for i, pred in enumerate(predictions) if i not in empty_indexes]
    references = [ref for i, ref in enumerate(references) if i not in empty_indexes]
    paths = [path for i, path in enumerate(paths) if i not in empty_indexes]
    scores = []
    with open(f"{current_split}_predictions.txt", "w+", encoding="utf-8") as f:
        for path, prediction, reference in zip(paths, predictions, references):
            score = None
            try:
                score = metric.compute(predictions = [prediction], references = [reference])
                scores.append(score)
                score = round(score, 6)
            except:
                print(traceback.print_exc())
                pass
            f.write(f"{path} :: {prediction} :: {reference} :: {score}\n")
    
    

    logger.info(f"""
                ***** eval metrics *****
                    eval_samples      : {len(predictions)}
                    eval_{cur_metric} : {float(sum(scores)) / int(len(scores))}
                    eval_runtime      : {time() - start_time} 
                """)
    
    
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine tuned model dir. relative dir path, or repo_id from huggingface")
    parser.add_argument("--load_script", help="script used for loading dataset for computing metrics.")
    parser.add_argument("--lang", help="ko ja zh en")
    args = parser.parse_args()
    main(args)
