import datasets
import csv
import os


_DESCRIPTION = """
Description needed.
"""
_HOMEPAGE = "cliveworks.co.kr"
#Metadata file
#audio TAR archive at
_DATA_URL = "https://drive.google.com/uc?export=download&id=1mykdXKesl9VmR58-k4AOmHbaOxkpSh1-"
# _DATA_URL = "https://www.dropbox.com/scl/fi/58fhlnhgtwjqthz47yjq8/audio.tar.gz?rlkey=vqk0elbjw5ue29fgcddtvcyae&dl=1"

class CliveDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "sentence": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
        )
    # Download and define the dataset splits
    def _split_generators(self, dl_manager):
        """Returns SplitGenerator"""
        # prompts_path = dl_manager.download(_PROMPTS_URL)
        audio_path = dl_manager.download(_DATA_URL)
        local_extracted_archive = dl_manager.extract(audio_path) if not dl_manager.is_streaming else None
        splits = {
            "train": "https://drive.google.com/uc?export=download&id=1xa_mOB9iCQmD57BE8Q8ctaX5zKTLcyez",
            "test": "https://drive.google.com/uc?export=download&id=1T860iEg8Zf5QYHv3i0E4Un-jSMZwAZ7c",
            "validation": "https://drive.google.com/uc?export=download&id=1u_MeOxrC2dl1OsEZg7To-Yfid6qbZpIK"
        }
        # splits = {
        #     "train": "https://www.dropbox.com/scl/fi/efflncphlhvtm0m5v2l73/train_split.tar?rlkey=ket7su2lundqm69e621ijc5ph&dl=1",
        #     "test": "https://www.dropbox.com/scl/fi/y2z6n75tag8wv1ja3rk1x/test_split.tar?rlkey=s61vz4eisr34f7jcx343ok7a6&dl=1",
        #     "validation": "https://www.dropbox.com/scl/fi/sfplm1oddjkfi91xsq8t8/validation_split.tar?rlkey=1j6vzjpuveek8waigx1ptxw6f&dl=1"
        # }
        
        metadata_paths = {}
        for split in splits:
            extracted_path = dl_manager.download_and_extract(splits[split])
            metadata_paths[split] = os.path.join(extracted_path, f"{split}.tsv")
        
        return [
            datasets.SplitGenerator(
                name = datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "audio_files": dl_manager.iter_archive(audio_path),
                    "metadata_path": metadata_paths["train"]
                }
            ),
            datasets.SplitGenerator(
                name = datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "audio_files": dl_manager.iter_archive(audio_path),
                    "metadata_path": metadata_paths["test"]
                }
            ),
            datasets.SplitGenerator(
                name = datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "audio_files": dl_manager.iter_archive(audio_path),
                    "metadata_path": metadata_paths["validation"]
                }
            )
        ]
    
    
    
    def _generate_examples(self, audio_files, local_extracted_archive, metadata_path):
        """Yields examples as (key, example) tuples."""
        metadata = {"path": [], "sentence": []}
        with open(metadata_path, "r", encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                audio_path, transcription = row[0].strip().split(" :: ")
                metadata["path"].append(audio_path)
                metadata["sentence"].append(transcription)
        
        id_ = 0 
        for path, f in audio_files:
            if path in metadata["path"]:    
                result = {}
                path = os.path.join(local_extracted_archive, path) if local_extracted_archive else audio_path
                result["path"] = path
                result["sentence"] = transcription
                result["audio"] = {"path": audio_path, "bytes": f.read()}
                yield id_, result
                id_ += 1  

    