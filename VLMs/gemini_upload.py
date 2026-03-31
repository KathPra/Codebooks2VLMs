from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os
import time

load_dotenv()
API_KEY_ = os.getenv("GEMINI_API_KEY")

# default: Load the model
client = genai.Client(api_key = API_KEY_)
"""
# List all models
for model in client.models.list():
    # Check if 'batchGenerateContent' is in the supported actions
    if 'batchGenerateContent' in model.supported_actions:
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Input Limit: {model.input_token_limit}")
        print("-" * 30)

for model in client.models.list():
    # We check if the batch method is in the supported_actions list
    supports_batch = 'batchGenerateContent' in model.supported_actions
    status = "✅ YES" if supports_batch else "❌ No"
    print(f"{model.name:<40} | {status}")
"""
status_json = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/status_gemini.jsonl"
batch_inputs_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/"
res_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_outputs/"
output_file_path = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/"

# check existing
existing = []
with open(status_json,"r") as f:
    for line in f:
        data = json.loads(line)
        for id_, val in data.items():
            name = val["name"]
            existing.append(name)
print(existing)


versions = ["org", "alt_1", "alt_2"]
categories = ["animals", "climateaction", "consequences"]#, "setting"]#, "type"]
datasets = ["tv"]#,"ct"]

for data in datasets:
    for c in categories:
        for v in versions:
            d = f"{data}_gemini-3.1-flash-lite-preview_{c}_{v}"
            # for tv only process "org"


            # check if A/B/C split files exist for this combination
            splits = [""]
            if os.path.exists(f"{batch_inputs_path}batch_{d}_C.jsonl"):
                splits = ["_C"]
            elif os.path.exists(f"{batch_inputs_path}batch_{d}_A.jsonl"):
                splits = ["_A", "_B",""]

            for split in splits:
                d_new = d + split
                print(d_new, flush=True)

                # skip if already uploaded
                if d_new in set(existing):
                    continue

                # skip if already parsed (e.g. was processed as a non-split batch previously)
                if os.path.exists(res_path + f"results_{d_new}.jsonl"):
                    print(d_new, "already parsed, skipping upload.", flush=True)
                    continue

                uploaded_file = client.files.upload(
                    file=f"{batch_inputs_path}batch_{d_new}.jsonl",
                    config=types.UploadFileConfig(display_name=d_new, mime_type='jsonl')
                )
                file_batch_job = client.batches.create(
                    model="models/gemini-3.1-flash-lite-preview",
                    src=uploaded_file.name,
                    config={
                        'display_name': d_new,
                    },
                )

                with open(status_json, "a") as f:
                    f.write(json.dumps({file_batch_job.name: {"id": uploaded_file.name, "name": d_new}}) + "\n")
                
                time.sleep(1200)