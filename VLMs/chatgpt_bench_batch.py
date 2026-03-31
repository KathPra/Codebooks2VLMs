## conda env: openai
import os
import base64
import json
import tqdm 
import argparse
import timeit
import glob
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Set once globally

from utils import load_prompts, load_ct, load_tv, check_existing_batch, update_img_list, image2base64

def create_batch_file(output_file, img_path, id, cat, v, prompt, model_name):
    with open(output_file, "a") as f:
        _,img_b64 = image2base64(img_path)

        request = {
            "custom_id": f"{id}_{cat}_{v}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        ]
                    }
                ],
                "max_output_tokens": 16
            }
        }

        f.write(json.dumps(request) + "\n")


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
        chunks = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 26 chunks
        n_chunks = len(chunks)

        # check existing
        existing, check, all_done = check_existing_batch(f"{model_name}_{cat}", dataset, cat, len(img_paths))
        # if all prompts have been completed, skip to next category
        if len(check) == len(versions):
            print("all done.")
            continue

        # also load any already-written chunk entries to avoid duplicates on re-run
        for v in versions:
            for chunk in chunks:
                chunk_file = f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}_{chunk}.jsonl"
                if os.path.exists(chunk_file):
                    with open(chunk_file, "r") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                pid = data["custom_id"].split("_" + cat)[0]
                                existing[v].add(pid)

        # update image list to remove already processed images
        if len(all_done) > 0:
            img_paths = update_img_list(img_paths, all_done)
        # fix image paths for which the file extension is missing
        img_paths = [i.removesuffix(".jpg") + ".jpg" for i in img_paths]

        # create all chunk files upfront
        for v in versions:
            for chunk in chunks:
                chunk_file = f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}_{chunk}.jsonl"
                if not os.path.exists(chunk_file):
                    os.mknod(chunk_file)

        # iterate over images — assign each to a chunk based on its position
        total = len(img_paths)
        counter = 0
        for img_path in tqdm.tqdm(img_paths):
            img_id = img_path.split("/")[-1].split(".")[0]
            chunk = chunks[min(counter * n_chunks // total, n_chunks - 1)]

            for v in prompt_dict[cat]:
                # if all images have been processed for this prompt, skip to next prompt
                if v in check:
                    continue
                # process only if image has not been processed before
                if img_id not in existing[v]:
                    prompt = prompt_dict[cat][v]
                    create_batch_file(f"{base_path}batch_{dataset}_{model_save_name}_{cat}_{v}_{chunk}.jsonl", img_path, img_id, cat, v, prompt, model_name)
            counter += 1
                   

if __name__ == "__main__":
    main()

