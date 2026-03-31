import json
import tqdm
import glob
import os
from collections import defaultdict

res_paths = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_outputs/"
output_file_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/"

model_name = "gpt-5.4-mini"

all_res = glob.glob(res_paths + "results*" + model_name + "*.jsonl")

# Group result files by their base name (stripping chunk suffix _A through _R)
# e.g. results_tv_gpt-5.4-mini_animals_org_C.jsonl maps to
#      results_tv_gpt-5.4-mini_animals_org.jsonl
groups = defaultdict(list)
for res in all_res:
    basename = os.path.basename(res)
    name_no_ext = basename[:-6]  # strip ".jsonl"
    if name_no_ext[-2] == "_" and name_no_ext[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        base_key = name_no_ext[:-2] + ".jsonl"
    else:
        base_key = basename
    groups[base_key].append(res)

for base_name, res_files in groups.items():
    # determine dataset
    if "_ct_" in base_name:
        dataset = "ct"
    else:
        dataset = "tv"

    # extract version and category from the base filename
    # e.g. results_tv_gpt-5.4-mini_animals_org.jsonl -> tv_gpt-5.4-mini_animals_org
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

    # process all source files for this group (sorted A→Z)
    with open(out_path, "a") as out_f:
        for res in sorted(res_files):
            with open(res, "r") as f:
                for line in tqdm.tqdm(f, desc=os.path.basename(res)):
                    data = json.loads(line)
                    id_ = data["custom_id"]

                    # extract prompt version
                    if "_alt_1" in id_:
                        lv = "alt_1"
                    elif "_alt_2" in id_:
                        lv = "alt_2"
                    elif "_org" in id_:
                        lv = "org"
                    else:
                        print("skipping", id_)
                        continue

                    # extract cat and img id
                    cat = id_.replace("_" + lv, "").split("_")[-1]
                    img_id = id_.replace("_" + cat + "_" + lv, "")

                    # skip if already parsed
                    if img_id in existing_preds:
                        continue

                    # extract results
                    preds = data["response"]
                    if preds is None:
                        continue
                    rtext = preds["body"]["output"][0]["content"][0]["text"]

                    # extract runtime
                    try:
                        created = preds["body"]["created_at"]
                        completed = preds["body"]["completed_at"]
                        dur = completed - created
                    except Exception:
                        dur = "na"

                    out_f.write(json.dumps({img_id: {"pred": rtext, "time": dur}}) + "\n")
                    existing_preds.add(img_id)
