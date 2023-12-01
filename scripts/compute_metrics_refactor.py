import argparse
import torch
import librosa
from tqdm import tqdm
import logging
import sys
import re
from time import time
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
        self.special_chars = self.get_special_chars_regex(args.lang)
        self.empty_files = []
        self.references = []
        self.predictions = []

        self.raw_dataset = load_dataset(args.load_script, split="validation")
        self.raw_dataset = self.raw_dataset.map(self.remove_special_characters, num_proc=8, desc="remove special chars")
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

    def get_special_chars_regex(self, lang):
        chars_to_ignore = {
            "ko": re.compile("[.?!.,]"),
            "ja": re.compile("[+-。、「」&#8203;``【oaicite:0】``&#8203;〜〉…？?.,~!]"),
            "zh": re.compile("[。？，！.,?~]"),
            "en": re.compile("[.,?!~]")
        }
        return chars_to_ignore.get(lang, re.compile(""))

    def remove_special_characters(self, batch):
        batch["target_text"] = re.sub(self.special_chars, "", batch["target_text"])
        return batch

    def speech_file_to_array_fn(self, batch):
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

    def run_recognizer(self):
        start_time = time()
        vectorized_dataset = self.raw_dataset.map(
            self.speech_file_to_array_fn,
            num_proc=8,
            desc="preprocess datasets"
        )

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

    def postprocess_predictions(self):
        with open("empty_files.txt", "w+", encoding="utf-8") as f:
            for path in self.empty_files:
                f.write(f"{path}\n")

        self.logger.info("***** Simple postprocessing *****")
        for i, pair in tqdm(enumerate(zip(self.predictions_temp, self.references_temp))):
            prediction, reference = pair
            prediction = prediction.strip()
            prediction = "".join(list(prediction)).lower()
            reference = reference.strip()
            reference = "".join(list(reference)).lower()
            if len(prediction) > 0 and len(reference) > 0:
                self.predictions.append(prediction)
                self.references.append(reference)
            else:
                self.logger.warning(f"Prediction or reference is empty")

        with open("predictions.txt", "w+", encoding="utf-8") as f:
            for prediction, reference in zip(self.predictions, self.references):
                f.write(f"{prediction} :: {reference}\n")

        with open("samples_metrics.txt", mode="w+", encoding="utf-8") as f:
            for pred, ref in zip(self.predictions, self.references):
                cer = evaluate.load("cer").compute(predictions=[pred], references=[ref])
                wer = evaluate.load("wer").compute(predictions=[pred], references=[ref])
                f.write(f"{pred} :: {ref} :: cer={cer} :: wer={wer}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine-tuned model dir. relative dir path, or repo_id from huggingface")
    parser.add_argument("--load_script", help="script used for loading dataset for computing metrics.")
    parser.add_argument("--lang", help="ko ja zh en")
    args = parser.parse_args()
    
    #run speech recognizer
    recognizer = SpeechRecognizer(args)
    recognizer.run_recognizer()
    # write empty files first
    with open(f"empty_files_{args.lang}", mode="w+", encoding="utf-8") as f:
        for empty_file_name in recognizer.empty_files:
            f.write(f"{empty_file_name}")

    t = time.localtime()
    current_time = time.strftime("%H%M%S", t)
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    with open(f"sample_metrics_{args.lang}_{current_time}", mode="w+", encoding="utf-8") as g:
        for pred, ref in zip(recognizer.predictions, recognizer.references):
            cer = cer_metric.compute(predictions=[pred], references=[ref])
            wer = wer_metric.compute(predictions=[pred], references=[ref])
            f.write(f"{pred} :: {ref} :: cer={cer} :: wer={wer}\n")
      
