## conda env: openai
import os
import json
import tqdm 
import argparse
from google.genai import types
from dotenv import load_dotenv
from utils import load_prompts, load_ct, load_tv, check_existing_batch, update_img_list, image2base64

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Set once globally

def create_batch_file(output_file, img_path, id, cat, v, prompt, model_name):
    with open(output_file, "a") as f:
        _, image_str = image2base64(img_path)

        request = {
            "key": f"{id}_{cat}_{v}",
            "request": {
                "contents":[{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_str
                                }
                            }
                        ]
                    }]
                }
            }


        f.write(json.dumps(request) + "\n")


#####################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ct","tv"])
    parser.add_argument("--cat", type=str, required=False, choices=["animals", "climateaction", "consequences", "setting", "type"])
    args = parser.parse_args()

    model_name = "gemini-3.1-flash-lite-preview"

    dataset = args.dataset
    if args.cat:
        req_cat = [args.cat]
    else:
        req_cat = ["animals", "climateaction", "consequences", "setting", "type"]      

    ######################################

    print("Using model:", model_name, " for benchmarking on dataset ", dataset,".", flush=True)
    # set base path
    base_path = f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/"
    model_save_name = model_name.replace("/","_")

    device = "cuda:0"

    prompt_dict = load_prompts()

    if dataset == "ct":
        img_paths = load_ct()


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
        # check existing (fix arg order: full_save_name, dataset, cat, num)
        existing, check, all_done = check_existing_batch(f"{model_save_name}_{cat}", dataset, cat, len(img_paths))
        # if all prompts have been completed, skip to next category
        if len(check) == len(versions):
            print("all done.")
            continue

        # update image list to remove already processed images
        if len(all_done) > 0:
            img_paths = update_img_list(img_paths, all_done)
        
        print("Processing", len(img_paths), "images.")

        # fix image paths for which the file extension is missing
        img_paths = [i.removesuffix(".jpg") + ".jpg" for i in img_paths]

        # create output files upfront
        for v in versions:
            for split in ["_A", "_B"]:
                ab_file = f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}{split}.jsonl"
                if not os.path.exists(ab_file):
                    os.mknod(ab_file)

        # iterate over images — counter tracks position for A/B split
        counter = 0
        half = len(img_paths) // 2
        for img_path in tqdm.tqdm(img_paths):
            img_id = img_path.split("/")[-1].split(".")[0]

            #iterate over prompt versions
            for v in prompt_dict[cat]:
                # if all images have been processed for this prompt, skip to next prompt
                if v in check:
                    continue
                # process only if image has not been processed before
                if img_id not in existing[v]:
                    # load prompt
                    prompt = prompt_dict[cat][v]
                    if counter < half:
                        create_batch_file(f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}_A.jsonl", img_path, img_id, cat, v, prompt, model_name)
                    else:
                        create_batch_file(f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}_B.jsonl", img_path, img_id, cat, v, prompt, model_name)
                    
                    # hacky fix for missing type (org) images
                    #create_batch_file(f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}_C.jsonl", img_path, img_id, cat, v, prompt, model_name)

            counter += 1  # one increment per image, outside the version loop
                   

if __name__ == "__main__":
    main()

