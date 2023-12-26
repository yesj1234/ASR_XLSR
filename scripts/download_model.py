from huggingface_hub import snapshot_download
import argparse
import os

def main(args):
    REPO_ID = args.repo_id
    LOCATION_DIR = args.location_dir
    if not os.path.exists(LOCATION_DIR):
        os.mkdir(LOCATION_DIR)
    snapshot_download(repo_id = REPO_ID,
                    local_dir_use_symlinks=False,
                    local_dir = LOCATION_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--location_dir", required=True)
    args = parser.parse_args()
    main(args)