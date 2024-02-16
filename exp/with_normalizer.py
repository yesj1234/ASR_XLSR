import neologdn 
from fugashi import Tagger 
import jaconv
import evaluate 
import pandas as pd 
from tqdm import tqdm 
import pprint
class Normalizer:
    def __init__(self):
        self.fugashi_tagger = Tagger('-Owakati') 
    
    def call_origin(self, text):
        return text
    
    def call_fugashi(self, text):
        return self.fugashi_tagger.parse(text)
    
    def call_neologdn(self, text):
        return neologdn.normalize(text)  
    
    def call_jaconv(self, text):
        return jaconv.normalize(text, "NFKC")
    
    def to_hiragana(self, text): 
        return jaconv.kata2hira(text)
    



# import gc 
def main(args):
    normalizer = Normalizer()
    LIB_KEYS = ['ORIGIN', 'NEOLOGDN', 'FUGASHI', 'JACONV']
    COLUMNS = ["PREDICTION", "REFERENCE", "SCORE"]
    FUNC_CALL = {
        "ORIGIN": normalizer.call_origin,
        "NEOLOGDN": normalizer.call_neologdn,
        "FUGASHI": normalizer.call_fugashi,
        "JACONV": normalizer.call_jaconv
    }
    pd_dict = {
        k1: {
            k2: [] for k2 in COLUMNS
        } for k1 in LIB_KEYS
    }
    
    metric = evaluate.load("cer")
    with open(args.prediction_file) as f: 
        lines = f.readlines()
        count = 0        
        for line in tqdm(lines, desc="Looping", ascii=" =", leave=True):
            _, pred, ref, score = line.split(" :: ")
            # score 3 times. origin, fugashi, neologdn 

            for k in LIB_KEYS:
                cur_pred = FUNC_CALL[k](pred)
                cur_ref = FUNC_CALL[k](ref)
                score = (metric.compute(predictions=[cur_pred], references=[cur_ref]))
                for col in COLUMNS:
                    if col == "PREDICTION":
                        pd_dict[k][col].append(cur_pred)
                     
                    if col == "REFERENCE":
                        pd_dict[k][col].append(cur_ref)
                            
                    if col == "SCORE":
                        pd_dict[k][col].append(score)
                                   
            # if count > 10:
            #     break 
            # count += 1
    
    df = pd.DataFrame()
    for k in LIB_KEYS:
        for col in COLUMNS:
            df.insert(loc=len(df.columns), column=f"{k}_{col}", value=pd_dict[k][col]) 
    df.to_csv("with_normalization.csv", index=False)



if __name__ == "__main__":
    import argparse 
    parser = argparse.add_argument("--prediction_file")
    args = parser.parse_args()
    main(args)
  
