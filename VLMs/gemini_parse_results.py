import json
import tqdm
import glob
import os
from collections import defaultdict

res_paths = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_outputs/"
output_file_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/"

model_name = "gemini-3.1-flash-lite-preview"
all_res = glob.glob(res_paths + "results*" + model_name + "*.jsonl")
print(all_res)

# Group result files by their base name (stripping _A/_B suffix)
# e.g. results_tv_gemini-..._animals_org_A.jsonl and _B both map to
#      results_tv_gemini-..._animals_org.jsonl
groups = defaultdict(list)
for res in all_res:
    basename = os.path.basename(res)
    base_key = basename
    for suffix in ["_A.jsonl", "_B.jsonl", "_C.jsonl"]:
        if basename.endswith(suffix):
            base_key = basename[: -len(suffix)] + ".jsonl"
            break
    groups[base_key].append(res)

for base_name, res_files in groups.items():
    # determine dataset
    if "_ct_" in base_name:
        dataset = "ct"
    else:
        dataset = "tv"

    # extract version and category from the base filename
    # e.g. results_tv_gemini-..._animals_org.jsonl -> tv_gemini-..._animals_org
    stripped = base_name.replace("results_", "").replace(".jsonl", "")
    if "_alt_1" in stripped:
        v = "alt_1"
    elif "_alt_2" in stripped:
        v = "alt_2"
    elif "_org" in stripped:
        v = "org"
    else:
        print("unknown version, skipping", base_name)
        continue

    # cat is the token between model and version suffix
    cat_part = stripped.replace(f"{dataset}_{model_name}_", "").replace(f"_{v}", "")

    curr = f"{dataset}_{model_name}_final_{cat_part}_{v}"
    out_path = f"{output_file_path}{curr}.jsonl"

    # count total entries across all source files for this group
    total_source = 0
    for res in res_files:
        with open(res, "r") as f:
            total_source += sum(1 for line in f if line.strip())

    # load already-parsed img_ids from the output file
    existing_preds = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as ef:
            for line in ef:
                if line.strip():
                    existing_preds.update(json.loads(line).keys())

    print(f"{curr}: {len(existing_preds)}/{total_source} parsed")
    if len(existing_preds) >= total_source:
        print(f"  -> fully done, skipping")
        continue

    # process all source files (A, B, or unsplit) for this group
    with open(out_path, "a") as out_f:
        for res in sorted(res_files):  # sort so A before B
            with open(res, "r") as f:
                for line in tqdm.tqdm(f, desc=os.path.basename(res)):
                    data = json.loads(line)
                    id_ = data["key"]

                    # extract prompt version
                    if "_org" in id_:
                        lv = "org"
                    elif "_alt_1" in id_:
                        lv = "alt_1"
                    elif "_alt_2" in id_:
                        lv = "alt_2"
                    else:
                        print("skipping", id_)
                        continue

                    # extract cat and img id
                    cat = id_.replace("_" + lv, "").split("_")[-1]
                    img_id = id_.replace("_" + cat + "_" + lv, "")

                    # skip if already parsed
                    if img_id in existing_preds:
                        continue

                    existing_preds.add(img_id)
                    
                    # if no response was created, print id and process later
                    if "response" not in data:
                        print(data["error"], data["key"])
                        continue
                    # handle blocked responses (e.g. PROHIBITED_CONTENT)
                    elif "candidates" not in data["response"]:
                        block_reason = data["response"].get("promptFeedback", {}).get("blockReason", "UNKNOWN")
                        print(f"blocked response ({block_reason}): {id_}")
                        rtext = f"BLOCKED:{block_reason}"
                        dur = "na"
                    else:
                        # extract results
                        preds = data["response"]["candidates"][0]["content"]
                        rtext = preds["parts"][0]["text"]

                        # extract runtime
                        try:
                            created = preds["body"]["created_at"]
                            completed = preds["body"]["completed_at"]
                            dur = completed - created
                        except Exception:
                            dur = "na"

                    out_f.write(json.dumps({img_id: {"pred": rtext, "time": dur}}) + "\n")
                    
