import pandas as pd 
import json 
import pprint as pp 
import os
from tqdm import tqdm

DF_KEYS = ['contentsIdx', 'source', 'category', 'solved_copyright', 'origin_lang_type', 'origin_lang', 'contentsName', 'fi_source_filename', 'fi_source_filepath', 'li_platform_info', 'li_subject', 'li_location', 'fi_sound_filename', 'fi_sound_filepath', 'li_total_video_time', 'li_total_voice_time', 'li_total_speaker_num', 'fi_start_voice_time', 'fi_end_voice_time', 'fi_duration_time', 'tc_text', 'tl_trans_lang', 'tl_trans_text', 'tl_back_trans_lang', 'tl_back_trans_text', 'speaker_tone', 'sl_new_word', 'sl_abbreviation_word', 'sl_slang', 'sl_mistake', 'sl_again', 'sl_interjection', 'place', 'en_outside', 'en_insdie', 'day_night', 'en_day', 'en_night', 'speaker_name', 'speaker_gender_type', 'speaker_gender', 'speaker_age_group_type', 'speaker_age_group']
# function applied for the folder containing json files 
def create_df(path):
    df = pd.DataFrame(columns=DF_KEYS)
    for item in os.listdir(path):
        try: 
            with open(os.path.join(path, item), encoding="utf-8", mode="r") as jf:
                json_obj = json.load(jf)
                df.loc[len(df.index)] = json_obj       
        except Exception as e:
            print(e)
    return df

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the folder of folders")
    parser.add_argument("--csv_name", help="name of the saved csv file")
    args = parser.parse_args()
    pp.pprint(args)
    
    dataframes = []
    for root, folders, files in tqdm(os.walk(args.path), desc="lopping folders"):
        if files:
            dataframes.append(create_df(root))
    df = pd.concat(dataframes)
    df.to_csv(args.csv_name)
    
    
