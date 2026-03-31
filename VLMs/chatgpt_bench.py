## conda env: openai
import os
import base64
import json
import tqdm 
import argparse
import timeit
import glob
from dotenv import load_dotenv
from openai import OpenAI
import sys

from utils import load_prompts, load_ct, load_tv, check_existing, update_img_list, image2base64


load_dotenv()
API_KEY_ = os.getenv("CHATGPT_API_KEY")

#####################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ct","tv"])
    parser.add_argument("--cat", type=str, required=False, choices=["animals", "climateaction", "consequences", "setting", "type"])
    args = parser.parse_args()

    model_name = "gpt-5.4-mini"
    dataset = args.dataset
    if args.cat:
        req_cat = [args.cat]
    else:
        req_cat = ["animals", "climateaction", "consequences", "setting", "type"]      

    ######################################

    print("Using model:", model_name, " for benchmarking on dataset ", dataset,".", flush=True)
    # set base path
    base_path = f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/{dataset}"
    model_save_name = model_name.replace("/","_")

    device = "cuda:0"

    prompt_dict = load_prompts()

    if dataset == "ct":
        img_paths = load_ct()

    # default: Load the model 
    client = OpenAI(api_key=API_KEY_)

    # iterate through cat
    categories = ["animals", "climateaction", "consequences", "setting", "type"]
    versions = ["org", "alt_1", "alt_2"]
    for cat in categories:
        # process specific category only
        if cat not in req_cat:
            continue
        print("Processing ", cat)
        # if tv, load super-cat specific images
        if dataset == "tv":
            img_paths = load_tv(cat)
        # check existing
        existing, check, all_done = check_existing(f"{model_save_name}_final_{cat}", dataset, len(img_paths))
        # if all prompts have been completed, skip to next category
        if len(check) == len(versions):
            continue
        # update image list to remove already processed images
        if len(all_done) > 0:
            img_paths = update_img_list(img_paths, all_done)
        # iterate over images
        for img_path in tqdm.tqdm(img_paths):
            if img_path.__contains__(".jpg"):
                _,image_str = image2base64(img_path)
            else:
                _, image_str = image2base64(img_path+".jpg")

            img_id = img_path.split("/")[-1].split(".")[0]
            #iterate over prompt versions
            for v in prompt_dict[cat]:
                # if all images have been processed for this prompt, skip to next prompt
                if v in check:
                    continue
                # process only if image has not been processed before
                if img_id not in existing[v]:

                    # create output file
                    if not os.path.exists(f"{base_path}_{model_save_name}_final_{cat}_{v}.jsonl"):
                        os.mknod(f"{base_path}_{model_save_name}_final_{cat}_{v}.jsonl")

                    # load prompt
                    prompt = prompt_dict[cat][v]
                    try:
                        t1 = timeit.default_timer() # start timer
                        # interence
                        response = client.responses.create(
                            model = "gpt-5.4",
                            input=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": prompt
                                        },
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:image/jpeg;base64,{image_str}"
                                        }
                                    ]
                                }
                            ],
                            max_output_tokens=16
                        )
                        rtext = response.output_text
                        t2 = timeit.default_timer() # stop timer
                        dur = t2-t1
                        
                        # save prediction to file
                        with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/{dataset}_{model_save_name}_final_{cat}_{v}.jsonl", "a") as f:
                            f.write(json.dumps({img_id:{"pred": rtext, "time":dur}}) + "\n")
                    
                        # print to slurm out
                        print(f"{img_id} - {cat} ({dur} sec): {rtext}", flush=True)
                        exit()

                    except Exception as e:
                        print(f"Failed on {img_id}: {e}")


if __name__ == "__main__":
    main()

