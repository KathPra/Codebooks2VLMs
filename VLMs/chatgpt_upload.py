## conda env: openai
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import time

status_json = "/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/status_chatgpt.jsonl"

load_dotenv()
API_KEY_ = os.getenv("CHATGPT_API_KEY")
client = OpenAI(api_key=API_KEY_)

# check existing and drain any in-progress batches before submitting new ones
existing = []
in_progress_ids = []
with open(status_json,"r") as f:
    for line in f:
        data = json.loads(line)
        for id_, val in data.items():
            name = val["name"]
            existing.append(name)
            status = client.batches.retrieve(id_).status
            if status in ("failed","cancelled"):
                existing.remove(name) 
            elif status not in ("completed", "cancelled", "expired"):
                print(f"In-progress batch found: {name} ({id_}) — {status}")
                in_progress_ids.append(id_)

print(len(in_progress_ids), "batches are currently running", flush=True)

if in_progress_ids:
    print(f"Waiting for {len(in_progress_ids)} in-progress batch(es) to finish before submitting new ones...")
    while in_progress_ids:
        time.sleep(60)
        in_progress_ids = [
            id_ for id_ in in_progress_ids
            if client.batches.retrieve(id_).status not in ("completed", "failed", "cancelled", "expired")
        ]
        print(f"  Still waiting on {len(in_progress_ids)} batch(es)...")

print(f"Quota clear. Existing submissions: {existing}")

versions = ["org", "alt_1", "alt_2"]
categories = ["animals", "climateaction", "consequences", "setting", "type"]
datasets = ["tv","ct"]

for data in datasets:
    for c in categories:
        for v in versions:
            # create output name
            d = f"{data}_gpt-5.4-mini_{c}_{v}"
            if data == "tv":
                # process only org
                if v != "org":
                    continue
                # iterate over all chunks A–R
                for s in ["_" + c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]:
                    d_new = d + s

                    print(d_new, flush=True)
                    if d_new not in set(existing):
                        batch_input_file = client.files.create(
                            file=open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/batch_{d_new}.jsonl", "rb"),
                            purpose="batch"
                        )

                        print(batch_input_file.id)

                        batch = client.batches.create(
                            input_file_id=batch_input_file.id,
                            endpoint="/v1/responses",
                            completion_window="24h"
                        )

                        print(batch.id)

                        with open(status_json, "a") as f:
                            f.write(json.dumps({batch.id :{"id": batch_input_file.id, "name": d_new}})+ "\n")
                        # poll until batch completes before submitting the next chunk
                        # (avoids exceeding the 1M enqueued token limit)
                        while True:
                            status = client.batches.retrieve(batch.id).status
                            print(f"  {d_new}: {status}", flush=True)
                            if status in ("completed", "failed", "cancelled", "expired"):
                                break
                            time.sleep(60)






            ## process ct
            else:
                print(d)
                if d not in set(existing):
                    batch_input_file = client.files.create(
                        file=open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/batch_inputs/batch_{d}.jsonl", "rb"),
                        purpose="batch"
                    )

                    print(batch_input_file.id)

                    batch = client.batches.create(
                        input_file_id=batch_input_file.id,
                        endpoint="/v1/responses",
                        completion_window="24h"
                    )

                    print(batch.id)

                    with open(status_json, "a") as f:
                        f.write(json.dumps({batch.id :{"id": batch_input_file.id, "name": d}})+ "\n")
                    # poll until batch completes before submitting the next one
                    while True:
                        status = client.batches.retrieve(batch.id).status
                        print(f"  {d}: {status}", flush=True)
                        if status in ("completed", "failed", "cancelled", "expired"):
                            break
                        time.sleep(60)