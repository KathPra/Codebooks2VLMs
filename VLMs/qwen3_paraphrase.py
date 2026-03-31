## conda env: qwenner
import os
from PIL import Image, ImageFile
import json
import tqdm 
import argparse
import timeit

# allow loading oversize images
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["HF_HUB_CACHE"] = "/ceph/lprasse/home2/huggingface/hub"
os.environ["HF_HOME"] = "/ceph/lprasse/home2/huggingface/"
os.environ["HF_DATASETS_CACHE"] = "/ceph/lprasse/home2/huggingface/datasets"

from transformers import AutoProcessor

#####################
def load_prompts(prompt_type):
    prompts = []
    with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/prompt_cache/Qwen-Qwen3-VL-8B-Instruct_{prompt_type}.jsonl", "r") as f:
        for i in f:
            prompts.append(json.loads(i))
    prompt_dict = {}
    for p in prompts:
        #extract data
        cat = list(p.keys())[0]
        id_ = list(p[cat].keys())[0]
        prompt = p[cat][id_]["prompt"]
        # add to output
        if cat not in prompt_dict:
            prompt_dict[cat]={}
        prompt_dict[cat][id_]= prompt

    return prompt_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, choices=["long", "short"])
    parser.add_argument("--model_name", type=str, choices=["Qwen/Qwen3-VL-8B-Instruct",  "Qwen/Qwen3-VL-30B-A3B-Instruct"])
    parser.add_argument("--prompt_ablation", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_concepts_supercat", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_think", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_localization", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", type=str, required=False, choices=["ct","tv"])
    args = parser.parse_args()

    prompt_type = args.prompt_type
    model_name = args.model_name
    prompt_ablation = args.prompt_ablation
    use_concepts_supercat = args.use_concepts_supercat
    use_think = args.use_think
    use_localization = args.use_localization

    #model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    model_name = "Qwen/Qwen3-VL-8B-Instruct" # high cpu mem: 60gb

    if prompt_ablation:
        dataset = "abl"
    else:
        dataset = args.dataset

    prompt_save_name = prompt_type +"_"+ ("concept-supercat" if use_concepts_supercat else "no-concept") + "_" + ("loc" if use_localization else "no-loc") + "_" + ("think" if use_think else "no-think")
    print(prompt_save_name)

    ######################################
    print("Using model:", model_name, "with", prompt_type, "prompts", flush=True)
    # set base path
    base_path = f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/{dataset}"
    model_save_name = model_name.replace("/","_")

    # check if completed
    if os.path.exists(base_path+f"{base_path}_{model_save_name}_{prompt_save_name}_animals_org.jsonl"):
        print("Already processed. Skipping.")
        exit(1)

    # load prompts and paraphrases
    prompt_dict = load_prompts(prompt_save_name)

    device = "cuda:0"

    # default: Load the model on the available device(s)
    if model_name == "Qwen/Qwen3-VL-30B-A3B-Instruct":
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype="auto", device_map="auto",
            #attn_implementation="flash_attention_2"
        )

    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left" # Required for batch generation

    # folder containing images
    with open("/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/scripts/prompt_ablation_sample.json", "r") as f:
        img_list = json.load(f)
    print(len(img_list), flush=True)

    # prep output
    # iterate through images
    for img_path in tqdm.tqdm(img_list):
        if img_path.__contains__(".jpg"):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.open(img_path+".jpg").convert("RGB")
        img_id = img_path.split("/")[-1].split(".")[0]
    # check if output files exist, otherwise create
        for cat in prompt_dict:
            for v in prompt_dict[cat]:
                # create output file
                if not os.path.exists(f"{base_path}_{model_save_name}_{prompt_save_name}_{cat}_{v}.jsonl"):
                    os.mknod(f"{base_path}_{model_save_name}_{prompt_save_name}_{cat}_{v}.jsonl")

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
                with open(f"/ceph/lprasse/ClimateVisions/ClimateVisions_2.0/analysis/vlm_pred/{dataset}_{model_save_name}_{prompt_save_name}_{cat}_{v}.jsonl", "a") as f:
                    f.write(json.dumps({img_id:{"pred": pred, "time":dur}}) + "\n")
            
                # print to slurm out
                print(f"{img_id} - {cat} ({dur} sec): {pred}", flush=True)

if __name__ == "__main__":
    main()
