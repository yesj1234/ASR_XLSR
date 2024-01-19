import os 
import shutil 
import logging 
import sys 
logger = logging.getLogger("CopySplit2")
logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.INFO,
                    handlers= [logging.StreamHandler(sys.stdout)])
def main(args):
    
    # 1. check if destination folders exists 
    if args.do_copy_wav:
        if not os.path.exists(args.wav_destination):
            os.makedirs(args.wav_destination)
 
    txt_dir_path = "/".join(args.txt_destination.split("/")[:-1])
    if not os.path.exists(txt_dir_path):
        os.makedirs(txt_dir_path)
        
    with open(args.source, mode="r", encoding="utf-8") as f, open(args.txt_destination, mode="w", encoding="utf-8") as t:
        lines = f.readlines()
        count = 0 
        scores = []
        new_lines = []
        for line in lines: 
            audio, prediction, reference, score = line.split(" :: ")
            # /home/ubuntu/3.보완조치완료/3.Test/1.원천데이터/2.중국어/패션,뷰티/11958/11958_981197_355.00_357.00.wav
            score = float(score[:-2])
            audio_filename = audio.split("/")[-1]
            if score < float(args.max_score):
                new_lines.append((audio, prediction, reference, score))
                if args.do_copy_wav:
                    shutil.copyfile(src=audio, dst=os.path.join(args.wav_destination, audio_filename))
                count += 1
                scores.append(score)
        new_lines.sort(key=lambda x: x[0])
        for new_line in new_lines:
            t.write(f"{new_line[0]} :: {new_line[1]} :: {new_line[2]} :: {new_line[3]}\n")    
    logging.info(f"""
                filename: {args.txt_destination}
                total: {len(lines)}
                copied: {count}
                score_mean: {sum(scores) / count}
                 """)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="source txt file containing generated predictions and references")
    parser.add_argument("--max_score", default=0.5)
    parser.add_argument("--wav_destination")
    parser.add_argument("--txt_destination")
    parser.add_argument("--do_copy_wav", default="")
    args = parser.parse_args()
    main(args)