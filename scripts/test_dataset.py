import os
import datasets 
import re

from datasets.download.download_manager import DownloadManager

_DESCRIPTION = "data for computing CER / WER metrics."

class SampleSpeech(datasets.GeneratorBasedBuilder): 
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000)
                }
            )
        )
    
    """Returns SplitGenerators."""
    VERSION = datasets.Version("0.0.1")
    def _split_generators(self, dl_manager: DownloadManager):
        # self.data_dir = os.environ["DATA_DIR"]
        # self.audio_dir = os.environ["AUDIO_DIR"]
        
        self.data_dir = os.path.join("/home/data/최종검증/dataset/asr_split")
        self.audio_dir = os.path.join("/home/data/최종검증/dataset/")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "test_refined.tsv"),
                    "split": "test"
                }
            )
        ]    
    def _generate_examples(self, filepath, split): 
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding='utf-8') as f:
            data = f.read().strip()
            for id_, row in enumerate(data.split("\n")):
                path, sentence = tuple(row.split(" :: "))
                if os.path.exists(os.path.join(self.audio_dir, path)):
                    with open(os.path.join(self.audio_dir, path), 'rb') as audio_file:
                        audio_data = audio_file.read()
                    audio = {
                        "path": os.path.join(self.audio_dir, path),
                        "bytes": audio_data,
                        "sampling_rate": 16_000
                    }
                    
                    yield id_, {
                        "file": os.path.join(self.audio_dir, path),
                        "audio": audio,
                        "target_text": sentence,
                    }
                