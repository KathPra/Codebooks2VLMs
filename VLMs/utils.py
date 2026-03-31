import json
import os
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO
import base64

def load_prompts():
    prompt_dict = {}
    # load results of prompt ablation
    with open("/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/prompts/vlm_prompt_types.json","r") as f:
        prompt_choice = json.load(f)
    # load prompt for each category
    for cat in prompt_choice:
        prompt_type = prompt_choice[cat]
        print("loading ", prompt_type, "for super-category ", cat, ".")
        prompts = []
        with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/prompt_cache/Qwen-Qwen3-VL-8B-Instruct_{prompt_type}.jsonl", "r") as f:
            for i in f:
                prompts.append(json.loads(i))
        for p in prompts:
            #extract data
            cat = list(p.keys())[0]
            id_ = list(p[cat].keys())[0]
            prompt = p[cat][id_]["prompt"]
            # add to output
            if cat not in prompt_dict:
                prompt_dict[cat]={}
            if id_ in ["org", "alt_1", "alt_2"]:
                prompt_dict[cat][id_]= prompt

    return prompt_dict


def load_ct():
    folder = "/ceph/lprasse/ClimateVisions/Images/ClimateCT/*.jpg"
    img_list = glob.glob(folder)
    print(len(img_list), flush=True)
    return img_list


def load_tv(cat):
    with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/annotation/qualitymatch_confirmed/{cat}.json", "r") as f:
        cat_imgs = json.load(f)
    folder = "/ceph/lprasse/ClimateVisions/Images/ClimateTV_cleaned/"
    month_dict = {"01": "01_January","02": "02_February","03": "03_March",
                "04": "04_April","05": "05_May","06": "06_June",
                "07": "07_July","08": "08_August","09": "09_September",
                "10": "10_October","11": "11_November","12": "12_December"}
    cat_imgs_full = []
    for i in cat_imgs:
        timestamp = i.split("_")[-1]
        year, month, _ = timestamp.split("-")
        month = month_dict[month]
        cat_imgs_full.append(folder+year+"/"+month+"/"+i)
    cat_imgs = None
    return cat_imgs_full


def check_existing(full_save_name, dataset, num):
    # check if output files exist, otherwise create
    existing = {}
    check = []
    all_done = set()
    for v in ["org", "alt_1", "alt_2"]:
        base_path = f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/{dataset}"
        if os.path.exists(f"{base_path}_{full_save_name}_{v}.jsonl"):
            pid_keys = set()
            with open(f"{base_path}_{full_save_name}_{v}.jsonl","r") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        # Get the first (and only) key safely
                        pid_keys.add(next(iter(data)))
            existing[v] = pid_keys
            if len(pid_keys) == num:
                check.append(v)
            elif len(pid_keys) > num:
                print("Output file contains error: too many predictions")
        else:
            existing[v] = set()
        # create set of image ids which have predictions for all prompts
        if len(all_done) == 0:
                all_done = existing[v]
        else:
            all_done = all_done & existing[v]
        
    # sanity check
    all_len = [len(existing[x]) for x in existing]
    len_ad = len(all_done)
    if min(all_len) < len_ad:
        raise ValueError("set of completed images is incorrect.")
    return existing, check, all_done


def check_existing_batch(full_save_name, dataset, cat, num):
    # check if output files exist, otherwise create
    existing = {}
    check = []
    all_done = set()
    for v in ["org", "alt_1", "alt_2"]:
        base_path = f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/batch_{dataset}"
        pid_keys = set()
        counter = 0
        for split in ["_A", "_B", "_C", "_D", "","_E", "_F", "_G", "_H", "_I", "_J", "_K","_L","_M","_N","_O","_P","_Q","_R","_S","_T","_U","_V","_W","_X","_Y","_Z"]:
            split_file = f"{base_path}_{full_save_name}_{v}{split}.jsonl"
            if not os.path.exists(split_file):
                continue
            with open(split_file, "r") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        if "gpt-5.4-mini" in full_save_name:
                            pid_keys.add(data["custom_id"])
                        elif "gemini-3.1-flash-lite-preview" in full_save_name:
                            pid_keys.add(data["key"])
                        counter += 1
        if counter == 0:
            print("file not found")
            existing[v] = set()
        else:
            if counter != len(pid_keys):
                print("output contains duplicates!!!")
            pid_keys = list(pid_keys)
            pid_keys = [p.split("_"+cat)[0] for p in pid_keys]
            existing[v] = set(pid_keys)
            if len(pid_keys) == num:
                check.append(v)
            elif len(pid_keys) > num:
                print("Output file contains error: too many predictions")
        # create set of image ids which have predictions for all prompts
        if len(all_done) == 0:
                all_done = existing[v]
        else:
            all_done = all_done & existing[v]
        
    # sanity check
    all_len = [len(existing[x]) for x in existing]
    len_ad = len(all_done)
    if min(all_len) < len_ad:
        raise ValueError("set of completed images is incorrect.")
    return existing, check, all_done

def update_img_list(img_list, all_done):
    new_list = []
    for path_str in img_list:
        # Extract just the filename (e.g., 'report_01.csv') from the full path
        file_name = Path(path_str).name.split(".")[0]
        
        if file_name not in all_done:
            new_list.append(path_str)
    return new_list


def smart_resize(img, max_pixels=512*512):
    w, h = img.size
    if w * h <= max_pixels:
        return img
    
    scale = (max_pixels / (w * h)) ** 0.5
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

def image2base64(img_path, max_pixels=512*512, quality=85):
    with Image.open(img_path) as img:
        img = img.convert("RGB")

        # 🔹 adaptive resize
        img = smart_resize(img, max_pixels=max_pixels)

        # 🔹 convert to bytes (in memory)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)

        image_bytes = buffer.getvalue()  # <-- THIS is raw bytes

        # 🔹 optional: base64 encode (for API)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return image_bytes, image_base64