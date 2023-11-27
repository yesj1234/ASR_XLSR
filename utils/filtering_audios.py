import librosa
import os
import logging 
import argparse
logger = logging.getLogger(__name__)
logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.INFO)
def main(args):
    folder = args.wav_dir
    logger.info(folder)
    count = 0
    wavs = []
    for root, dirs, files in os.walk(folder):
        if files:
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext == ".wav":
                    current_wav = os.path.join(root, file)
                    duration = librosa.get_duration(filename = current_wav)
                    if duration == 0:
                        count += 1
                        wavs.append(os.path.join(root, file))
                        logger.info(f"""
                                    duration: {duration}
                                    filename: {current_wav}
                                    """)
    logger.info(f"files duration in seconds 0 : {count}")
    with open("no_sound_files.txt", mode="w+", encoding = "utf-8") as f:
        for row in wavs:
            f.write(f"{row}\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", help="folder path to filter wav files.")
    args = parser.parse_args()
    main(args)