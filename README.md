# ASR EXAMPLE

## PREPROCESSING 산출물 DATA

1. **_prepare_data.py_**

```bash
python3 prepare_data.py --asr_dest_folder /path/to/the/destination/folder --jsons /path/to/the/folder/containing/jsons
e.g.
python3 prepare_data.py --asr_dest_folder ./asr_split --jsons $SPLITS_DIR
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
## Dockerizing
### Prerequisites
Docker and NVIDIA driver MUST be installed in your host OS. 
1. Add Docker's official GPG key
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```
2. Add the repository to Apt sources:
```bash
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
3. verify the Docker Engine installation is succesful by running the hello-world image
```bash
sudo docker run hello-world
```
4. install docker nvidia(NVIDIA driver MUST be installed in the host OS)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
5. NVIDIA-docker install
```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
6. rerun docker service
```bash
sudo systemctl restart docker
```
7. test if nvidia docker is successfully installed
```bash
sudo docker run --rm --gpus all ubuntu:20.04 nvidia-smi
```
## Simply run the training scripts in 4 steps with docker. 
1. build 
```bash
sudo docker build -t xlsr .
```
2. run
```bash
bash run_container.sh # sudo docker run -it --ipc host --gpus all -v /home/ubuntu/data:/home/data -v /home/ubuntu/ASR_XLSR/scripts:/home/scripts xlsr bash
``` 
3. preprocess(inside the container)
```bash
python 1.prepare_from_json_asr.py --asr_dest_folder /home/data/한국어(KO)_일본어(JP) --jsons /home/data/한국어(KO)_일본어(JP) # python 1.prepare_from_json_asr.py --asr_dest_folder /home/data/SourceLang(lang_code)_TargetLang(lang_code)
python refine_data.py --tsv_splits_dir /home/data/한국어(KO)_일본어(JP)/asr_split # python refine_data.py --tsv_splits_dir /home/data/SourceLang(lang_code)_TargetLang(lang_code)/asr_split
```
4. change DATA_DIR, AUDIO_DIR in run_speech_recognition_ctc.sh and run training
```bash
export DATA_DIR=/home/data/'한국어(KO)_일본어(JP)'/asr_split # Change this to the actual path
export AUDIO_DIR=/home/data/
```
```bash
bash run_speech_recognition_ctc.sh
```
# ENVIRONMENT
- OS: Canonical Ubuntu 20.04 
- CPU: 64 OCPU(Oracle CPU)
- Memory: 16GB(per GPU)
- Storage: 7.68TB NVMe SSD Storage(x2)
- GPU: NVIDIA A10(x4)
