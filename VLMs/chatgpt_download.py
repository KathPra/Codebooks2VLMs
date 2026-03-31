from openai import OpenAI
from dotenv import load_dotenv
import json
import os
load_dotenv()
API_KEY_ = os.getenv("CHATGPT_API_KEY")
client = OpenAI(api_key=API_KEY_)

status_json = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/status_chatgpt.jsonl"

output_file_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/"

verbose = False

# check existing
with open(status_json,"r") as f:
    for line in f:
        data = json.loads(line)
        for id_, val in data.items():
            name = val["name"]

            # extract chunk suffix if present (_A through _R)
            split = ""
            if len(name) >= 2 and name[-2] == "_" and name[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                split = name[-2:]
                name = name[:-2]

            # extract prompt version v
            if "_alt_1" in name:
                v = "_alt_1"
            elif "_alt_2" in name:
                v = "_alt_2"
            elif "_org" in name:
                v = "_org"
            else:
                print("unknown version, skipping", name)
                continue
            name = name.replace(v, "")
            v = v[1:]

            # extract dataset, model, and category
            dataset, model, cat = name.split("_")

            results_file = f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_outputs/results_{name}_{v}{split}.jsonl"

            # if results have been downloaded already, skip
            if os.path.exists(results_file):
                print(name, v, split, "already downloaded.")
                continue
            # if no A/B split and output file exists, fully parsed — skip
            if not split and os.path.exists(output_file_path + f"{dataset}_{model}_final_{cat}_{v}.jsonl"):
                print(name, v, "already parsed.")
                continue

            # retrieve status
            status = client.batches.retrieve(id_)

            print(name, v, split, status.status)
            # print model full status if verbose output is desired
            if verbose:
                if status.status == "failed":
                    print(json.dumps(status.model_dump(), indent=2))

            if status.error_file_id:
                error_file = client.files.content(status.error_file_id)
                with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_outputs/errors_{name}_{v}{split}.jsonl", "wb") as f:
                    f.write(error_file.content)

            if status.status == "completed":
                file_response = client.files.content(status.output_file_id)
                with open(results_file, "wb") as f:
                    f.write(file_response.content)