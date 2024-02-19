# Script for comparing different normalizers. 
import pandas as pd 
import argparse 
import json
import os 
import librosa 
import torch 
import evaluate
import pprint
from tqdm import tqdm
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from normalizers.english import EnglishTextNormalizer
from normalizers.basic import BasicTextNormalizer

def init(root):
    body = None
    for root, folders, files in os.walk(args.root):
        if files:
            for file in files:
                if os.path.splitext(file)[1] == '.json':
                    with open(os.path.join(root, file), mode='r', encoding='utf-8') as f:
                        json_obj = json.load(f)
                        body = {k: [] for k in json_obj.keys()}     
    return body

def get_data(json_path):
    with open(json_path, encoding='utf-8', mode="r") as f:
        json_obj = json.load(f)
        return json_obj

def format_audio_file_path(string):
    # expected original string be like: 'https://objectstorage.ap-seoul-1.oraclecloud.com/n/cnb97trxvnun/b/clive-resource/o/output/일본어_한국어/원천데이터/게임/7253/7253_24869_264.21_268.30.wav'
    # output should be something like: '/home/ubuntu/"3차 보완조치 2차"/"003 일본어"/3.Test/1.원천데이터/2.일본어/게임/7253/7253_24869_264.21_268.30.wav'
    path = string.split("/")[-5:]
    source_lang = path[0].split("_")[0]
    source_lang = f"1.{source_lang}" if source_lang == "한국어" else f"2.{source_lang}"  
    path.pop(0)
    path.insert(1, source_lang)
    path = '/'.join(path)
    path = "1." + path
    return path 

def main(args):
    #1. read json files under the given path(folder). 
    body = init(args.root)
    count = 0
    for root, folders, files in os.walk(args.root):
        flag = True 
        if files:
            for file in files:
                if os.path.splitext(file)[1] == '.json':
                    data = get_data(os.path.join(root,file))
                    for k, v in body.items():
                        body[k].append(data[k])
                    if count >= args.max_count:
                        flag = False
                        break
                    count += 1
        if not flag:
            break 
    body['prediction'], body['score'] = [], []  
    if args.audio_path_prefix:
        body['fi_sound_filepath'] = list(map(format_audio_file_path, body['fi_sound_filepath']))
        body['fi_sound_filepath'] = list(map(lambda x: args.audio_path_prefix + x, body['fi_sound_filepath']))
    
    #3. generate prediction with the loaded audio and finetuned model. 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.language == "english":
        normalizer = EnglishTextNormalizer()
    if args.language in ["japanese", "chinese"]:
        normalizer = BasicTextNormalizer(split_letters=True)
    if args.language == "korean":
        normalizer = BasicTextNormalizer()
    
    metric = evaluate.load("wer")
    if args.model in ["yesj1234/enko_xlsr_100p_sup", "facebook/wav2vec2-large-xlsr-53"]:
        model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model)
        processor = Wav2Vec2Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
        count = 0
        for audio, tc in tqdm(zip(body['fi_sound_filepath'], body['tc_text']), desc="looping", ascii=" =", leave=True):
            y, sr = librosa.load(audio, sr=16000)
            try:
                inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
                logits = model(inputs.input_values, attention_mask = inputs.attention_mask).logits
                predicted_ids = torch.argmax(logits, axis = -1)
                predicted_sentence = processor.batch_decode(predicted_ids)[0]
                predicted_sentence = normalizer(predicted_sentence)
                body['prediction'].append(predicted_sentence)
            except Exception as e: 
                print(e)
                body['prediction'].append("NO PREDICTION GENERATED")
                                
            try:
                score = metric.compute(predictions=[predicted_sentence], references=[normalizer(tc)])
                score = round(score, 6)
            except Exception as e:
                score = -1
                print(e)
                
            body['score'].append(score)
            if count >= args.max_count:
                break
            count += 1
        
    if args.model in ["openai/whisper-large-v2", "openai/whisper-large-v3"]:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
        processor = WhisperProcessor.from_pretrained(args.model)
        model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
        count = 0
        for audio, tc in tqdm(zip(body['fi_sound_filepath'], body['tc_text']), desc="looping", ascii=" =", leave=True):
            y, sr = librosa.load(audio, sr=16000)
            try:
                input_features = processor(y, sampling_rate = 16_000, return_tensors="pt").input_features.to(device)
                predicted_ids = model.generate(input_features)
                predicted_sentence = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                predicted_sentence = normalizer(predicted_sentence)
                body['prediction'].append(predicted_sentence)
            except Exception as e:
                print(e)
                body['prediction'].append("NO PREDICTION GENERATED")
                
            try:
                score = metric.compute(predictions=[predicted_sentence], references=[normalizer(tc)])
                score = round(score, 6)
            except Exception as e:
                score = -1
                print(e)
            body['score'].append(score)
            if count >= args.max_count:
                break
            count += 1
    df = pd.DataFrame(body)
    
    df.to_csv(f"{args.language}_{args.model.replace('/', '.')}.csv", index=False)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="folder path containing json files")
    parser.add_argument("--audio_path_prefix")
    parser.add_argument("--model", choices=["yesj1234/enko_xlsr_100p_sup", "facebook/wav2vec2-large-xlsr-53", "openai/whisper-large-v2", "openai/whisper-large-v3"], type=str)
    parser.add_argument("--language", choices=["english", "korean", "japanese", "chinese"], type=str)
    parser.add_argument("--max_count", default=200000, type=int)
    args = parser.parse_args()
    main(args)
