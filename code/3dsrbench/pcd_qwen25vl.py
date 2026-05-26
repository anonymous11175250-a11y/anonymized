import os
import json
import argparse
import base64
import re
from io import BytesIO

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCD on 3DSRBench orientation tasks"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["3B", "7B", "32B"],
        help="Model size to use (3B, 7B, or 32B)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Alpha parameter for contrastive decoding"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/3dsrbench_v1_vlmevalkit_circular.tsv",
        help="Path to the TSV data file"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../../outputs/3dsrbench",
        help="Root directory for outputs"
    )
    return parser.parse_args()


def decode_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def is_valid_option(x):
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    return str(x).strip() != ""


def build_options(row):
    options = {}
    for k in ["A", "B", "C", "D"]:
        if k in row and is_valid_option(row[k]):
            options[k] = str(row[k]).strip()
    return options, list(options.keys())


def options_to_text(options_dict):
    return "\n".join([f"{k}. {v}" for k, v in options_dict.items()])


LEFT_RIGHT_EXTRACTION_PROMPT = """\
Extract the "reference object" and "target object" from a spatial reasoning question.
- reference object: the object whose position and facing direction defines the viewpoint
- target object: the object whose left/right position is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the chair's position facing where it is facing, is the desk on the left or right of me?
A: {"ref_obj": "chair", "target_obj": "desk"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the green car's position facing where it is facing, is the traffic cone on the left or right of me?
A: {"ref_obj": "green car", "target_obj": "traffic cone"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the person's position facing where it is facing, is the bicycle on the left or right of me?
A: {"ref_obj": "person", "target_obj": "bicycle"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the woman in pink cloth's position facing where it is facing, is the man holding a bag on the left or right of me?
A: {"ref_obj": "woman in pink cloth", "target_obj": "man holding a bag"}

Q: {question}
A:"""


FRONT_BEHIND_EXTRACTION_PROMPT = """\
Extract the "reference object" and "target object" from a spatial reasoning question.
- reference object: the object whose position and facing direction defines the viewpoint
- target object: the object whose front/behind position is being asked about
Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:
Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the table's position facing where it is facing, is the bookshelf in front of me or behind me?
A: {"ref_obj": "table", "target_obj": "bookshelf"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the truck's position facing where it is facing, is the gray car in front of me or behind me?
A: {"ref_obj": "truck", "target_obj": "stop sign"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the man's position facing where it is facing, is the backpack in front of me or behind me?
A: {"ref_obj": "man", "target_obj": "gray car"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the girl in red dress's position facing where it is facing, is the boy with glasses in front of me or behind me?
A: {"ref_obj": "girl in red dress", "target_obj": "boy with glasses"}

Q: {question}
A:"""


@torch.no_grad()
def extract_objects_via_llm(question, category, model, processor, device):
    if category == "orientation_on_the_left":
        prompt_template = LEFT_RIGHT_EXTRACTION_PROMPT
    else:
        prompt_template = FRONT_BEHIND_EXTRACTION_PROMPT
    
    prompt = prompt_template.replace("{question}", question.strip())

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt"
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

    generated_text = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    try:
        json_match = re.search(r'\{[^{}]*\}', generated_text)
        if json_match:
            parsed = json.loads(json_match.group())
            ref_obj = str(parsed.get("ref_obj", "object")).strip()
            target_obj = str(parsed.get("target_obj", "object")).strip()
            if ref_obj and target_obj:
                return ref_obj, target_obj
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return "object", "object"


def build_rewrite_q(ref_obj, target_obj, category):
    if category == "orientation_on_the_left":
        return f"Is the {target_obj} on the left or right of the {ref_obj}?"
    else:
        return f"is the {target_obj} in front of the {ref_obj} or behind the {ref_obj}?"


def extract_question_body(question):
    prefix_patterns = [
        "Consider the real-world 3D locations and orientations of the objects. ",
        "Consider the real-world 3D locations and orientations of the objects.\n",
    ]
    
    question_body = question
    for pattern in prefix_patterns:
        if question.startswith(pattern):
            question_body = question[len(pattern):]
            break
    
    return question_body


def build_original_prompt(question, options_text):
    prefix = "Consider the real-world 3D locations and orientations of the objects."
    question_body = extract_question_body(question)
    return f"""Question: {prefix} {question_body}

Options:
{options_text}

Return only the correct option letter.
"""


def build_camera_prompt(rewrite_q, options_text):
    prefix = "Consider the real-world 3D locations and orientations of the objects."
    return f"""Question: {prefix} From the camera view, {rewrite_q}

Options:
{options_text}

Return only the correct option letter.
"""


def build_viewpoint_check_prompt(ref_obj, category):
    if category == "orientation_on_the_left":
        return (
            f"Consider the real-world 3D locations and orientations of the objects.\n"
            f"Question:\n"
            f"In image, are you viewing the {ref_obj}'s back side?\n"
            f"Answer only yes or no."
        )
    else:
        return (
            f"Question: Consider the real-world 3D locations and orientations of the objects."
            f"In image, are you looking at the {ref_obj} from behind?"
            f"Answer only yes or no."
        )


def build_model_inputs(image, prompt, processor, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    return inputs


@torch.no_grad()
def generate_answer(image, prompt, model, processor, device):
    inputs = build_model_inputs(image, prompt, processor, device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

    generated_text = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    return generated_text


@torch.no_grad()
def check_same_viewpoint(image, ref_obj, category, model, processor, device):
    prompt = build_viewpoint_check_prompt(ref_obj, category)
    raw = generate_answer(image, prompt, model, processor, device)
    
    if category == "orientation_on_the_left":
        is_same = raw.lower().startswith("yes")
    else:
        is_same = raw.lower().startswith("no")
    
    return raw, is_same


def get_candidate_token_ids_for_label(label, tokenizer):
    variants = [label, " " + label, "\n" + label]

    ids = set()
    for v in variants:
        tok = tokenizer.encode(v, add_special_tokens=False)
        if len(tok) == 1:
            ids.add(tok[0])

    if not ids:
        tok = tokenizer.encode(label, add_special_tokens=False)
        ids.add(tok[-1])

    return sorted(ids)


@torch.no_grad()
def score_next_token_logprobs(image, prompt, model, processor, device):
    inputs = build_model_inputs(image, prompt, processor, device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    logp = F.log_softmax(logits, dim=-1)[0]
    return logp


def contrastive_pick_option(image, original_prompt, camera_prompt, allowed_labels, 
                            alpha, model, processor, device, option_token_ids):
    orig_logp = score_next_token_logprobs(image, original_prompt, model, processor, device)
    cam_logp = score_next_token_logprobs(image, camera_prompt, model, processor, device)

    scores = {}
    for label in allowed_labels:
        ids = option_token_ids[label]

        orig_best = max(orig_logp[i].item() for i in ids)
        cam_best = max(cam_logp[i].item() for i in ids)

        scores[label] = {
            "orig": orig_best,
            "cam": cam_best,
            "contrastive": orig_best - alpha * cam_best
        }

    pred = max(scores.items(), key=lambda x: x[1]["contrastive"])[0]
    return pred, scores


def process_category(df_category, category, model, processor, device, alpha, 
                     model_name_safe, output_dir, option_token_ids):
    results = []
    logs = []
    
    extraction_cache_path = os.path.join(
        output_dir,
        f"{model_name_safe}_{category}_extraction_cache.json"
    )
    
    viewpoint_cache_path = os.path.join(
        output_dir,
        f"{model_name_safe}_{category}_viewpoint_cache.json"
    )
    
    if os.path.exists(extraction_cache_path):
        with open(extraction_cache_path, "r") as f:
            extraction_cache = json.load(f)
    else:
        extraction_cache = {}
    
    if os.path.exists(viewpoint_cache_path):
        with open(viewpoint_cache_path, "r") as f:
            viewpoint_cache = json.load(f)
    else:
        viewpoint_cache = {}
    
    for idx in tqdm(range(len(df_category)), desc=f"Processing {category}"):
        row = df_category.iloc[idx]
        qid = str(row["qid"])
        question = str(row["question"])

        image = decode_image(row["image"])
        options_dict, labels = build_options(row)
        options_text = options_to_text(options_dict)

        if qid in extraction_cache:
            ref_obj = extraction_cache[qid]["ref_obj"]
            target_obj = extraction_cache[qid]["target_obj"]
        else:
            ref_obj, target_obj = extract_objects_via_llm(
                question, category, model, processor, device
            )
            extraction_cache[qid] = {"ref_obj": ref_obj, "target_obj": target_obj}
            with open(extraction_cache_path, "w") as f:
                json.dump(extraction_cache, f, indent=2, ensure_ascii=False)

        rewrite_q = build_rewrite_q(ref_obj, target_obj, category)
        
        original_prompt = build_original_prompt(question, options_text)
        camera_prompt = build_camera_prompt(rewrite_q, options_text)

        if qid in viewpoint_cache:
            vp_raw = viewpoint_cache[qid]["raw"]
            is_same_vp = viewpoint_cache[qid]["is_same"]
        else:
            vp_raw, is_same_vp = check_same_viewpoint(
                image, ref_obj, category, model, processor, device
            )
            viewpoint_cache[qid] = {"raw": vp_raw, "is_same": is_same_vp}
            with open(viewpoint_cache_path, "w") as f:
                json.dump(viewpoint_cache, f, indent=2, ensure_ascii=False)

        if is_same_vp:
            pred = generate_answer(image, camera_prompt, model, processor, device)
            pred_letter = pred.strip()[0].upper() if pred.strip() else "A"
            if pred_letter not in labels:
                pred_letter = labels[0]
            pred = pred_letter
        else:
            pred, _ = contrastive_pick_option(
                image=image,
                original_prompt=original_prompt,
                camera_prompt=camera_prompt,
                allowed_labels=labels,
                alpha=alpha,
                model=model,
                processor=processor,
                device=device,
                option_token_ids=option_token_ids
            )

        answer = str(row["answer"]).strip()
        hit = int(pred == answer)

        results.append([
            idx,
            str(row["qid"]),
            str(row["question"]),
            options_dict.get("A", ""),
            options_dict.get("B", ""),
            options_dict.get("C", ""),
            options_dict.get("D", ""),
            pred,
            str(row["category"]),
            rewrite_q,
            pred,
            answer,
            hit
        ])

        log_entry = {
            "qid": qid,
            "category": str(row["category"]),
            "question": question,
            "prediction": pred,
            "answer": answer,
            "hit": hit
        }
        
        logs.append(log_entry)
    
    with open(extraction_cache_path, "w") as f:
        json.dump(extraction_cache, f, indent=2, ensure_ascii=False)
    
    with open(viewpoint_cache_path, "w") as f:
        json.dump(viewpoint_cache, f, indent=2, ensure_ascii=False)
    
    return results, logs


def main():
    args = parse_args()
    
    model_size_map = {
        "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
        "32B": "Qwen/Qwen2.5-VL-32B-Instruct"
    }
    
    MODEL_NAME = model_size_map[args.model_size]
    
    alpha_str = str(args.alpha).replace(".", "p")
    model_name_safe = f"PCD_Qwen2.5-VL-{args.model_size}-Instruct_alpha{alpha_str}"
    
    OUTPUT_DIR = os.path.join(args.output_root, model_name_safe)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Alpha: {args.alpha}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    print("Model loaded successfully!\n")
    
    option_token_ids = {
        label: get_candidate_token_ids_for_label(label, tokenizer)
        for label in ["A", "B", "C", "D"]
    }
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, sep="\t")
    
    categories = ["orientation_on_the_left", "orientation_in_front_of"]
    df_filtered = df[df["category"].isin(categories)].reset_index(drop=True)
    
    print(f"Total samples: {len(df_filtered)}")
    for cat in categories:
        count = len(df_filtered[df_filtered["category"] == cat])
        print(f"  - {cat}: {count}")
    print()
    
    all_results = []
    all_logs = []
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        df_category = df_filtered[df_filtered["category"] == category].reset_index(drop=True)
        results, logs = process_category(
            df_category=df_category,
            category=category,
            model=model,
            processor=processor,
            device=device,
            alpha=args.alpha,
            model_name_safe=model_name_safe,
            output_dir=OUTPUT_DIR,
            option_token_ids=option_token_ids
        )
        
        all_results.extend(results)
        all_logs.extend(logs)
        
        cat_acc = sum(log["hit"] for log in logs) / len(logs) if logs else 0
        print(f"Accuracy: {cat_acc:.4f}")
    
    excel_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name_safe}_3DSRBenchv1_openai_result.xlsx"
    )
    
    json_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name_safe}_results.json"
    )
    
    for i, result in enumerate(all_results):
        result[0] = i
    
    pd.DataFrame(all_results, columns=[
        "index", "qid", "question", "A", "B", "C", "D",
        "prediction", "category", "col9", "col10", "answer", "hit"
    ]).to_excel(excel_path, index=False)
    
    with open(json_path, "w") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    
    total = len(all_logs)
    overall_acc = sum(log["hit"] for log in all_logs) / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Overall accuracy  : {overall_acc:.4f}")
    
    for category in categories:
        cat_logs = [log for log in all_logs if log["category"] == category]
        cat_acc = sum(log["hit"] for log in cat_logs) / len(cat_logs) if cat_logs else 0
        print(f"  - {category}: {cat_acc:.4f}")
    
    print(f"{'='*60}")
    print(f"Saved Excel  → {excel_path}")
    print(f"Saved JSON   → {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()