## conda env: qwenner
import os
from PIL import Image, ImageFile
import json
import tqdm 
import argparse
import timeit
import glob
import torch

from utils import load_prompts, load_ct, load_tv, check_existing, update_img_list

# allow loading oversize images
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["HF_HUB_CACHE"] = "/ceph/lprasse/home2/huggingface/hub"
os.environ["HF_HOME"] = "/ceph/lprasse/home2/huggingface/"
os.environ["HF_DATASETS_CACHE"] = "/ceph/lprasse/home2/huggingface/datasets"

from transformers import AutoProcessor

#####################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["google/gemma-3-4b-it"])
    parser.add_argument("--dataset", type=str, choices=["ct","tv"])
    parser.add_argument("--cat", type=str, required=False, choices=["animals", "climateaction", "consequences", "setting", "type"])
    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    if args.cat:
        req_cat = args.cat
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

    # default: Load the model on the available device(s)
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, 
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )


    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
                image = Image.open(img_path).convert("RGB")
            else:
                image = Image.open(img_path+".jpg").convert("RGB")
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
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {"type": "text", 
                                "text": prompt},
                            ],
                        }
                    ]

                    # Preparation for inference
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(device)

                    # Inference: Generation of the output
                    t1 = timeit.default_timer() # start timer
                    outputs = model.generate(**inputs, max_new_tokens=40)
                    pred = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
                    t2 = timeit.default_timer() # stop timer
                    dur = t2-t1
                    
                    # save prediction to file
                    with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/{dataset}_{model_save_name}_final_{cat}_{v}.jsonl", "a") as f:
                        f.write(json.dumps({img_id:{"pred": pred, "time":dur}}) + "\n")
                
                    # print to slurm out
                    print(f"{img_id} - {cat} ({dur} sec): {pred}", flush=True)

if __name__ == "__main__":
    main()
