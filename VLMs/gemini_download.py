from google import genai
from dotenv import load_dotenv
import json
import os
load_dotenv()
API_KEY_ = os.getenv("GEMINI_API_KEY")

# default: Load the model 
client = genai.Client(api_key = API_KEY_)

status_json = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/status_gemini.jsonl"
output_file_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/"

verbose = False

# check existing
batch_data = {}
with open(status_json,"r") as f:
    for line in f:
        data = json.loads(line)
        for id_, val in data.items():
            name = val["name"]

            # extract A/B split suffix if present
            split = ""
            for s in ["_A", "_B", "_C"]:
                if name.endswith(s):
                    split = s
                    name = name[: -len(s)]
                    break

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

            # prep readout
            batch_data[id_] = (name)
            # read out
            batch_job = client.batches.get(name=id_)

            print(name, v, split, batch_job.state.name)
            # print error if any
            if batch_job.state.name == "JOB_STATE_FAILED":
                print(batch_job.error)
            # download results if available
            if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
                result_file_name = batch_job.dest.file_name

                print("Downloading results")
                file_content = client.files.download(file=result_file_name)
                with open(results_file, "wb") as f:
                    f.write(file_content)
                        
