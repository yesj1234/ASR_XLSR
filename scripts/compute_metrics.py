import argparse
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer
)

from datasets import load_dataset 

def prepare_dataset(batch):
    # load audio
    sample = batch[audio_column_name]

    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(batch["input_values"])

    # encode targets
    additional_kwargs = {}
    if phoneme_language is not None:
        additional_kwargs["phonemizer_lang"] = phoneme_language

    batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
    return batch


def main():
    dataset = load_dataset("./sample_speech.py", split="test")
    
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_dir)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="fine tuned model dir. relative dir path, or repo_id from huggingface")
    args = parser.parse_args()
    main(args)
