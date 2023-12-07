import argparse
import torch
import librosa
from tqdm import tqdm
import logging
import sys
import re
import time 

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer
)
from datasets import load_dataset
import evaluate


class SpeechRecognizer:
    def __init__(self, args):
        self.args = args
        self.logger = self.setup_logger()
        self.empty_files = []
        self.references = []
        self.predictions = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(logging.INFO)
        return logger
    
    def load_dataset(self, loading_script):
        return load_dataset(loading_script)
    
    @staticmethod
    def speech_file_to_array_fn(batch):
        speech_array, _ = librosa.load(batch["file"], sr=16_000)
        batch["audio"] = speech_array
        batch["target_text"] = batch["target_text"]
        return batch

    def generate_predictions(self, batch):
        inputs = self.processor(batch["audio"], sampling_rate=16_000, return_tensors="pt", padding=True).to(
            self.device)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, axis=-1)
        predicted_sentences = self.processor.batch_decode(predicted_ids)
        return predicted_sentences

    def run_recognizer(self, vectorized_dataset):
        self.logger.info("***** Running Evaluation *****")
        for batch in tqdm(vectorized_dataset):
            try:
                predicted_sentence = self.generate_predictions(batch)
                predicted_sentence = predicted_sentence[0].strip()
                if len(predicted_sentence) < 1:
                    self.empty_files.append(batch["file"])
                else:
                    self.predictions.append(predicted_sentence)
                    self.references.append(batch["target_text"])
            except Exception as e:
                self.logger.warning(e)
                pass

def main(args):
    start = time.time()
    #run speech recognizer
    recognizer = SpeechRecognizer(args)
    raw_dataset = load_dataset(args.load_script)
    vectorized_dataset = raw_dataset.map(recognizer.speech_file_to_array_fn,
                                         num_proc = 8,
                                         desc="preprocess datasets")
    # run the inference 
    recognizer.run_recognizer(vectorized_dataset)
    
    # write empty files first
    with open(f"empty_files_{args.lang}", mode="w+", encoding="utf-8") as f:
        for empty_file_name in recognizer.empty_files:
            f.write(f"{empty_file_name}")
    
    # Simple post processing.
    recognizer.predictions = list(map(lambda x:x.strip(), recognizer.predictions)) # strip each samples 
    recognizer.references = list(map(lambda x:x.strip(), recognizer.references))
    recognizer.predictions = list(map(lambda x:"".join(list(x)).lower(), recognizer.predictions)) # remove multiple white spaces between words.
    recognizer.references = list(map(lambda x:"".join(list(x)).lower(), recognizer.references)) 
    recognizer.predictions = list(filter(lambda x:len(x) > 0, recognizer.predictions )) # filter out empty examples. 
    recognizer.references = list(filter(lambda x:len(x) > 0, recognizer.references )) # filter out empty examples. 


    # write each examples with each scores.
    t = time.localtime()
    current_time = time.strftime("%H%M%S", t)
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    with open(f"sample_metrics_{args.lang}_{current_time}.txt", mode="w+", encoding="utf-8") as g:
        for pred, ref in zip(recognizer.predictions, recognizer.references):
            cer = cer_metric.compute(predictions=[pred], references=[ref])
            wer = wer_metric.compute(predictions=[pred], references=[ref])
            f.write(f"{pred} :: {ref} :: cer={cer} :: wer={wer}\n")

    # score as a whole 
    total_cer = cer_metric.compute(predictions = recognizer.predictions, references = recognizer.references)
    total_wer = wer_metric.compute(predictions = recognizer.predictions, references = recognizer.references)
    recognizer.logger.info(f"""
    ***** eval metrics *****
    eval_samples: {len(recognizer.predictions)}
    eval_cer    : {total_cer}
    eval_wer    : {total_wer}
    eval_runtime: {time.time() - start} 
                           """)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine-tuned model dir. relative dir path, or repo_id from huggingface")
    parser.add_argument("--load_script", help="script used for loading dataset for computing metrics.")
    parser.add_argument("--lang", help="ko ja zh en")
    args = parser.parse_args()
    main(args)
    
