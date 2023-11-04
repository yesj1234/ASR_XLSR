import os 
import csv 
import evaluate

if __name__ == "__main__":
    predictions = []
    references = []
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    with open(os.path.join("predictions.txt"), "r+", encoding = "utf-8") as f:
        file= csv.reader(f, delimiter = "\n")
        for i, row in enumerate(file):
            prediction, reference = row[0].split(" :: ")
            prediction = prediction.strip()
            prediction = " ".join(list(prediction))
            reference = " ".join(list(reference))
            if len(prediction) > 0  and len(reference) >0:
                predictions.append(prediction)
                references.append(reference)
            else:
                print(f"prediction: {prediction}")
                print(f"reference: {reference}")
                pass 
            
    
    wer_score = wer.compute(predictions = predictions, references = references)
    cer_score = cer.compute(predictions = predictions, references = references)
    print(f"wer: {wer_score}")
    print(f"cer: {cer_score}")       
        
                