import os 
import csv 
import evaluate

if __name__ == "__main__":
    predictions = []
    references = []
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    with open(os.path.join("./validation_predictions.txt"), "r+", encoding = "utf-8") as f:
        file= csv.reader(f, delimiter = "\n")
        for i, row in enumerate(file):
            predicton = ""
            reference = ""
            try:
                _, prediction, reference, _ = row[0].split(" :: ")
            except ValueError:
                pass
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
    print(len(predictions))
    print(len(references))
            
    wer_score = wer.compute(predictions = predictions, references = references)
    cer_score = cer.compute(predictions = predictions, references = references)
    print(f"wer: {wer_score}")
    print(f"cer: {cer_score}")       
        
                
