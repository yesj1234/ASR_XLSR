import os 
import logging 
import sys 

logger = logging.getLogger("Combine_logger")
logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.INFO,
                    handlers= [logging.StreamHandler(sys.stdout)])
def main(args): 
    with open(args.file1, encoding="utf-8", mode="r") as f1, open(args.file2, encoding="utf-8", mode="r") as f2, open(args.destination, encoding="utf-8", mode="a+") as dest: 
        f1_lines = f1.readlines() 
        f2_lines = f2.readlines() 
        for line in f1_lines:
            dest.write(line)
        for line in f2_lines: 
            dest.write(line)
    logger.info(f"""
                {args.file1} : {len(f1_lines)}
                {args.file2} : {len(f2_lines)}
                total        : {len(f1_lines) + len(f2_lines)}
                
                """)

if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", help="file to combine")
    parser.add_argument("--file2")
    parser.add_argument("--destination", help="generated new split txt")
    args = parser.parse_args()
    
    main(args)