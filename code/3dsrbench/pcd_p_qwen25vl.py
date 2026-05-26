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
        description="Run PCD on 3DSRBench"
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
        default=1.0,
        help="Alpha parameter for contrastive decoding"
    )
    parser.add_argument(
        "--plausibility_alpha",
        type=float,
        default=0.001,
        help="Adaptive plausibility constraint"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=25,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/3dsrbench/3dsrbench_v1_vlmevalkit_circular.tsv",
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


def clean_excel_text(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)


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
Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the table's position facing where it is facing, is the bookshlef in front of me or behind me?
A: {"ref_obj": "table", "target_obj": "bookshelf"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the truck's position facing where it is facing, is the gray car in front of me or behind me?
A: {"ref_obj": "truck", "target_obj": "gray car"}

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the man's position facing where it is facing, is the doll in front of me or behind me?
A: {"ref_obj": "man", "target_obj": "doll"}

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


# =========================
# Input Builder
# =========================
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
def generate_answer(image, prompt, model, processor, device, max_new_tokens=50):
    inputs = build_model_inputs(image, prompt, processor, device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
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
    raw = generate_answer(image, prompt, model, processor, device, max_new_tokens=20)
    
    if category == "orientation_on_the_left":
        is_same = raw.lower().startswith("yes")
    else:
        is_same = raw.lower().startswith("no")
    
    return raw, is_same


@torch.no_grad()
def contrastive_generate_autoregressive(
    image, 
    original_prompt, 
    camera_prompt,
    cd_alpha, 
    plausibility_alpha, 
    model, 
    processor, 
    device,
    max_new_tokens=50
):
    inputs_orig = build_model_inputs(image, original_prompt, processor, device)
    inputs_cam = build_model_inputs(image, camera_prompt, processor, device)
    
    expert_input_ids = inputs_orig["input_ids"]
    amateur_input_ids = inputs_cam["input_ids"]
    
    generated_tokens = []
    
    eos_token_id = processor.tokenizer.eos_token_id
    
    for step in range(max_new_tokens):
        expert_outputs = model(**inputs_orig)
        expert_logits = expert_outputs.logits[:, -1, :] 
        
        amateur_outputs = model(**inputs_cam)
        amateur_logits = amateur_outputs.logits[:, -1, :]
        
        expert_logp = F.log_softmax(expert_logits, dim=-1)
        amateur_logp = F.log_softmax(amateur_logits, dim=-1)
        
        max_expert_logp = torch.max(expert_logp)
        log_plaus_alpha = torch.log(torch.tensor(plausibility_alpha, device=expert_logp.device))
        adaptive_threshold = log_plaus_alpha + max_expert_logp
        
        plausibility_mask = expert_logp >= adaptive_threshold
        
        contrastive_logp = expert_logp - cd_alpha * amateur_logp
        
        contrastive_logp_masked = contrastive_logp.clone()
        contrastive_logp_masked[~plausibility_mask] = float('-inf')
        
        if torch.all(torch.isinf(contrastive_logp_masked)):
            next_token_id = torch.argmax(expert_logp, dim=-1)
        else:
            next_token_id = torch.argmax(contrastive_logp_masked, dim=-1)
        
        next_token_id = next_token_id.item()
        
        if next_token_id == eos_token_id:
            break
        
        generated_tokens.append(next_token_id)
        
        expert_input_ids = torch.cat([
            expert_input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)
        
        amateur_input_ids = torch.cat([
            amateur_input_ids,
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)
        
        inputs_orig["input_ids"] = expert_input_ids
        inputs_orig["attention_mask"] = torch.ones_like(expert_input_ids)
        
        inputs_cam["input_ids"] = amateur_input_ids
        inputs_cam["attention_mask"] = torch.ones_like(amateur_input_ids)
    
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def extract_option_from_text(text, allowed_labels):
    if not text:
        return None
    
    text = text.strip()
    text_upper = text.upper()

    if text_upper in allowed_labels:
        return text_upper
    
    if len(text_upper) > 0 and text_upper[0] in allowed_labels:
        return text_upper[0]
    
    patterns = [
        r'(?:answer|option|choice)\s*(?:is|:|=)?\s*([A-D])',
        r'correct\s+(?:answer|option|choice)\s*(?:is|:|=)?\s*([A-D])',
        r'([A-D])\s+is\s+(?:the\s+)?(?:correct|right)',
        r'(?:选项|答案)\s*(?:是|：|:)?\s*([A-D])',
        r'\b([A-D])\s*[.。]?\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            option = match.group(1)
            if option in allowed_labels:
                return option
    
    for label in allowed_labels:
        if label in text_upper:
            return label
    
    return None


def process_category(df_category, category, model, processor, device, 
                     cd_alpha, plausibility_alpha, max_new_tokens,
                     model_name_safe, output_dir):
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
            generated_text = generate_answer(
                image, camera_prompt, model, processor, device, max_new_tokens
            )
        else:
            generated_text = contrastive_generate_autoregressive(
                image=image,
                original_prompt=original_prompt,
                camera_prompt=camera_prompt,
                cd_alpha=cd_alpha,
                plausibility_alpha=plausibility_alpha,
                model=model,
                processor=processor,
                device=device,
                max_new_tokens=max_new_tokens
            )

        pred = extract_option_from_text(generated_text, labels)
        if pred is None:
            pred = labels[0]

        answer = str(row["answer"]).strip()
        hit = int(pred == answer)

        results.append([
            idx,
            clean_excel_text(str(row["qid"])),
            clean_excel_text(str(row["question"])),
            clean_excel_text(options_dict.get("A", "")),
            clean_excel_text(options_dict.get("B", "")),
            clean_excel_text(options_dict.get("C", "")),
            clean_excel_text(options_dict.get("D", "")),
            clean_excel_text(pred),
            clean_excel_text(str(row["category"])),
            clean_excel_text(rewrite_q),
            clean_excel_text(generated_text),
            clean_excel_text(answer),
            hit
        ])

        log_entry = {
            "qid": qid,
            "category": str(row["category"]),
            "question": question,
            "options": options_dict,
            "generated_text": generated_text,
            "prediction": pred,
            "answer": answer,
            "correct": hit == 1
        }
        
        logs.append(log_entry)
    
    with open(extraction_cache_path, "w") as f:
        json.dump(extraction_cache, f, indent=2, ensure_ascii=False)
    
    with open(viewpoint_cache_path, "w") as f:
        json.dump(viewpoint_cache, f, indent=2, ensure_ascii=False)
    
    return results, logs


# =========================
# Main
# =========================
def main():
    args = parse_args()
    
    model_size_map = {
        "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
        "32B": "Qwen/Qwen2.5-VL-32B-Instruct"
    }
    
    MODEL_NAME = model_size_map[args.model_size]
    
    alpha_str = str(args.alpha).replace(".", "p")
    plaus_alpha_str = str(args.plausibility_alpha).replace(".", "p")
    
    model_name_safe = f"PCD_Qwen2.5-VL-{args.model_size}-Instruct_alpha{alpha_str}_plaus{plaus_alpha_str}"
    
    OUTPUT_DIR = os.path.join(args.output_root, model_name_safe)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"CD Alpha: {args.alpha}")
    print(f"Plausibility Alpha: {args.plausibility_alpha}")
    print(f"Device: {device}\n")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    df = pd.read_csv(args.data_path, sep="\t")
    
    categories = ["orientation_on_the_left", "orientation_in_front_of"]
    df_filtered = df[df["category"].isin(categories)].reset_index(drop=True)
    
    print(f"Total samples: {len(df_filtered)}\n")
    
    all_results = []
    all_logs = []
    
    for category in categories:
        df_category = df_filtered[df_filtered["category"] == category].reset_index(drop=True)
        results, logs = process_category(
            df_category=df_category,
            category=category,
            model=model,
            processor=processor,
            device=device,
            cd_alpha=args.alpha,
            plausibility_alpha=args.plausibility_alpha,
            max_new_tokens=args.max_new_tokens,
            model_name_safe=model_name_safe,
            output_dir=OUTPUT_DIR
        )
        
        all_results.extend(results)
        all_logs.extend(logs)
        
        cat_acc = sum(log["correct"] for log in logs) / len(logs) if logs else 0
        print(f"{category} accuracy: {cat_acc:.4f}")
    
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
    overall_acc = sum(log["correct"] for log in all_logs) / total if total > 0 else 0
    
    print(f"\nOverall accuracy: {overall_acc:.4f}")
    print(f"\nSaved Excel → {excel_path}")
    print(f"Saved JSON  → {json_path}\n")


if __name__ == "__main__":
    main()