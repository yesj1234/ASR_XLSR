# ASR EXAMPLE

## PREPROCESSING 산출물 DATA

0. **_validation_**

```bash
python3 0.json_validator.py --jsons /path/to/the/folder/containing/json/files
e.g.
python3 0.json_validator.py --jsons ./output
```

1. **_prepare_from_json_asr.py_**

```bash
python3 1.prepare_from_json_asr.py --asr_dest_folder /path/to/the/destination/folder --jsons /path/to/the/folder/containing/jsons
e.g.
python3 1.prepare_from_json_asr.py --asr_dest_folder ./asr_split --jsons $SPLITS_DIR
```

2. **_refine_data.py_**

Wav2Vec2 xls-r model

```bash
python3 refine_data.py --tsv_splits_dir /path/to/the/tsv/splits
e.g.
python3 refine_data.py --tsv_splits_dir ../asr_split
```

3. export the tsv file path and audio folder path for sample_speech.py to correctly load the data from local.

```bash
export DATA_DIR=/path/to/the/refined_splits
export AUDIO_DIR=/path/to/the/audio/folder
e.g.
export DATA_DIR=/home/ubuntu/my_asr/cycle0/asr_example/asr_split
export AUDIO_DIR=/home/ubuntu/output
```

4. set configurations before running the training script. possible arguments can be found in run_speech_recognition_ctc.sh. for example

```json
{
  "model_name_or_path": "facebook/wav2vec2-large-xlsr-53",
  "overwrite_output_dir": true,
  "freeze_feature_encoder": true,
  "attention_dropout": 0.1,
  "hidden_dropout": 0.1,
  "feat_proj_dropout": 0.1,
  "mask_time_prob": 0.3,
  "mask_feature_length": 64,
  "layerdrop": 0.1,
  "ctc_loss_reduction": "mean",
  "dataset_name": "./sample_speech.py",
  "train_split_name": "train",
  "audio_column_name": "audio",
  "text_column_name": "target_text",
  "eval_metrics": ["cer"],
  "unk_token": "[UNK]",
  "pad_token": "[PAD]",
  "word_delimiter_token": "|",
  "output_dir": "./ko-xlsr",
  "do_train": true,
  "do_predict": true,
  "evaluation_strategy": "steps",
  "eval_steps": 1000,
  "per_device_train_batch_size": 2,
  "per_device_eval_batch_size": 2,
  "gradient_accumulation_steps": 2,
  "num_train_epochs": 50,
  "save_strategy": "epoch",
  "logging_strategy": "epoch",
  "learning_rate": 5e-4,
  "warmup_steps": 500,
  "save_total_limit": 1,
  "group_by_length": true,
  "fp16": true,
  "max_duration_in_seconds": 10,
  "chars_to_ignore": [
    ",",
    "?",
    "!",
    "%",
    "'",
    "~",
    ":",
    "/",
    "(",
    ")",
    ".",
    "·",
    "\u001c",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "@"
  ]
}
```

5. run the training shell script

```bash
bash run_speech_recognition_ctc.bash
```

# Multi gpu 환경에서 trainer api 사용하기.

1. bash로 python script 실행 하기 혹은 그냥 python 명령어 실행.

```bash
LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node 4 \
--use-env run_training.py \
run_training_gpu_asr.json \
--chars_to_ignore [\,\?\.\!\-\;\:\"\“\‘\”\ ‘、。．！，・―─~｢｣『』〆｡\\\\※\[\]\{\}「」〇？…]
```
