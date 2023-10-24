from huggingface_hub import (
    login,
    HfApi
)
import os 
import argparse 

def main(args):    
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    api = HfApi()
    repo_url = api.create_repo(repo_id=args.repo_id, repo_type="model")
    if repo_url:
        api.upload_folder(
            folder_path=args.model_dir,
            repo_id = args.repo_id,
            repo_type = "model",
            token = os.environ["HUGGINGFACE_TOKEN"])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", help="huggingface repo id. yesj1234/[model_name]_cycle[#]_[lang]")
    parser.add_argument("--model_dir", help="folder path containing model checkpoint")
    parser.add_argument("--model_card", help="path to model card")
    args = parser.parse_args()
    main(args)
    