import os
import json
import argparse
import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCD on Omnispatial Allocentric tasks"
    )
    parser.add_argument("--model_size", type=str, required=True,
                        choices=["3B", "7B", "32B"])
    parser.add_argument("--alpha_lr", type=float, default=2.8,
                        help="Alpha parameter for left_right axis")
    parser.add_argument("--alpha_fb", type=float, default=1.4,
                        help="Alpha parameter for front_behind axis")
    parser.add_argument("--alpha_ab", type=float, default=1.0,
                        help="Alpha parameter for above_below axis")
    parser.add_argument("--dataset_path", type=str, default="../../data/omnispatial",
                        help="Path to Omnispatial dataset directory")
    parser.add_argument("--data_ids_path", type=str, 
                        default="../../data/omnispatial/data_ids.json",
                        help="Path to data_ids.json file")
    parser.add_argument("--output_root", type=str, default="../../outputs/omnispatial",
                        help="Root directory for outputs")
    parser.add_argument("--max_image_size", type=int, default=768)
    parser.add_argument("--group_index", type=int, default=0)
    parser.add_argument("--group", type=int, default=1)
    return parser.parse_args()


def resize_image_if_needed(image_path, max_size=768):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    if max(w, h) <= max_size:
        return image_path, img
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    temp_path = image_path.replace(".png", "_resized.png")
    img_resized.save(temp_path)
    return temp_path, img_resized


def categorize_options(options):
    axes = {}
    
    left_right_terms = ["left", "right"]
    front_behind_terms = ["front", "ahead", "forward", "rear", "behind", "backward"]
    above_below_terms = ["above", "below", "up", "down", "upward", "downward"]
    
    lr = []
    fb = []
    ab = []
    
    for opt in options:
        opt_lower = opt.lower()
        
        if "can not determine" in opt_lower or "cannot determine" in opt_lower:
            continue
        
        if any(term in opt_lower for term in left_right_terms):
            lr.append(opt)
        elif any(term in opt_lower for term in front_behind_terms):
            fb.append(opt)
        elif any(term in opt_lower for term in above_below_terms):
            ab.append(opt)
    
    if lr:
        axes["left_right"] = lr
    if fb:
        axes["front_behind"] = fb
    if ab:
        axes["above_below"] = ab
    
    return axes


EXTRACTION_PROMPTS = {
    "left_right": """\
Extract the "reference object" and "target object" from a spatial reasoning question about direction.
- reference object: the object whose position and facing direction defines the viewpoint
- target object: the object whose directional position (left/right) is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: If you stand at the position of the person and face the same direction as them, is the ball on your left or right?
A: {"ref_obj": "person", "target_obj": "ball"}

Q: From the swimmer's perspective, is the electronic board on their left or right?
A: {"ref_obj": "swimmer", "target_obj": "electronic board"}

Q: {question}
A:""",
    
    "front_behind": """\
Extract the "reference object" and "target object" from a spatial reasoning question about direction.
- reference object: the object whose position and facing direction defines the viewpoint
- target object: the object whose front/behind position is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: Standing at the athlete's position facing their direction, is the referee in front or behind?
A: {"ref_obj": "athlete", "target_obj": "referee"}

Q: If you were the person wearing the black hat and holding the bag in the image, where would the button be located relative to you?
A: {"ref_obj": "person wearing the black hat and holding the bag", "target_obj": "button"}

Q: {question}
A:""",
    
    "above_below": """\
Extract the "reference object" and "target object" from a spatial reasoning question about direction.
- reference object: the object whose upright orientation defines the viewpoint
- target object: the object whose above/below position is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: From the man's perspective, is the ball above or below him?
A: {"ref_obj": "man", "target_obj": "ball"}

Q: Where is the mirror located relative to the cup?
A: {"ref_obj": "cup", "target_obj": "mirror"}

Q: {question}
A:"""
}


@torch.no_grad()
def extract_objects_via_llm(question, options, axis_name, model, processor, device):
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    full_question = f"{question}\n\nOptions:\n{options_text}"
    
    prompt_template = EXTRACTION_PROMPTS.get(axis_name, EXTRACTION_PROMPTS["left_right"])
    prompt = prompt_template.replace("{question}", full_question)

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[text], padding=True, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
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


def build_rewrite_q(ref_obj, target_obj, axis_name):
    if axis_name == "left_right":
        return f"Is the {target_obj} on the left or right of the {ref_obj}?"
    elif axis_name == "front_behind":
        return f"is the {target_obj} in front of the {ref_obj} or behind the {ref_obj}?"
    else:
        return f"Is the {target_obj} above the {ref_obj} or below the {ref_obj}?"


def build_original_prompt(question, options):
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    
    return f"""Question: {question}

Options:
{options_text}

Return only the correct option letter."""


def build_camera_prompt(rewrite_q, options):
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    
    return f"""Question: From the camera view, {rewrite_q}

Options:
{options_text}

Return only the correct option letter."""


def build_viewpoint_check_prompt(ref_obj, axis_name):
    if axis_name == "left_right":
        return (
            f"Consider the real-world 3D locations and orientations of the objects.\n"
            f"Question:\n"
            f"In the image, are you looking at the {ref_obj}'s back side?\n"
            f"Answer only yes or no."
        )
    elif axis_name == "front_behind":
        return (
            f"Question: Consider the real-world 3D locations and orientations of the objects.In the image, are you looking at the {ref_obj} from behind?"
            f"Answer only yes or no."
        )
    else:
        return (
            f"Question: Consider the real-world 3D locations and orientations of the objects.In the image, is the {ref_obj} upside down (head at bottom, feet at top)? "
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
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    return inputs


@torch.no_grad()
def generate_answer(image, prompt, model, processor, device):
    inputs = build_model_inputs(image, prompt, processor, device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    generated_text = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    return generated_text


@torch.no_grad()
def check_same_viewpoint(image, ref_obj, axis_name, model, processor, device):
    prompt = build_viewpoint_check_prompt(ref_obj, axis_name)
    raw = generate_answer(image, prompt, model, processor, device)
    
    if axis_name == "left_right":
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
def get_next_token_logprobs(image, prompt, model, processor, device):
    inputs = build_model_inputs(image, prompt, processor, device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    logp = F.log_softmax(logits, dim=-1)[0]
    return logp


def magnitude_normalized_cd(
    image,
    original_prompt,
    axes_info,
    options,
    model,
    processor,
    device,
    option_token_ids,
    axis_alphas
):
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(options)]
    
    orig_logp = get_next_token_logprobs(image, original_prompt, model, processor, device)
    
    orig_scores = {}
    for i, opt in enumerate(options):
        label = labels[i]
        token_ids = option_token_ids[label]
        orig_scores[i] = max(orig_logp[tid].item() for tid in token_ids)
    
    final_scores = orig_scores.copy()
    
    for axis_name, info in axes_info.items():
        if not info["needs_cd"]:
            continue
        
        camera_prompt = info["camera_prompt"]
        cam_logp = get_next_token_logprobs(image, camera_prompt, model, processor, device)
        
        cam_scores = {}
        axis_option_indices = [options.index(opt) for opt in info["options"]]
        
        for idx in axis_option_indices:
            label = labels[idx]
            token_ids = option_token_ids[label]
            cam_scores[idx] = max(cam_logp[tid].item() for tid in token_ids)
        
        axis_alpha = axis_alphas[axis_name]
        
        cd_scores = {}
        for idx in axis_option_indices:
            cd_scores[idx] = orig_scores[idx] - axis_alpha * cam_scores[idx]
        
        orig_mean = np.mean([orig_scores[idx] for idx in axis_option_indices])
        cd_mean = np.mean([cd_scores[idx] for idx in axis_option_indices])
        shift = orig_mean - cd_mean
        
        for idx in axis_option_indices:
            final_scores[idx] = cd_scores[idx] + shift
    
    pred_idx = max(final_scores, key=final_scores.get)
    pred_letter = labels[pred_idx]
    
    return pred_letter


@torch.no_grad()
def direct_generation_single_axis(image, prompt, model, processor, device, options):
    raw = generate_answer(image, prompt, model, processor, device)
    
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(options)]
    
    pred_letter = raw.strip()[0].upper() if raw.strip() else "A"
    if pred_letter not in labels:
        pred_letter = "A"
    
    return pred_letter


def main():
    args = parse_args()
    
    axis_alphas = {
        "left_right": args.alpha_lr,
        "front_behind": args.alpha_fb,
        "above_below": args.alpha_ab
    }
    
    model_size_map = {
        "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
        "32B": "Qwen/Qwen2.5-VL-32B-Instruct"
    }
    
    MODEL_NAME = model_size_map[args.model_size]
    
    alpha_desc = f"lr{args.alpha_lr}_fb{args.alpha_fb}_ab{args.alpha_ab}"
    alpha_desc = alpha_desc.replace(".", "p")
    model_name_safe = f"PCD_Qwen2.5-VL-{args.model_size}_{alpha_desc}"
    
    OUTPUT_DIR = os.path.join(args.output_root, model_name_safe)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
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
        chr(65 + i): get_candidate_token_ids_for_label(chr(65 + i), tokenizer)
        for i in range(10)
    }
    
    with open(args.data_ids_path, 'r') as f:
        data_ids_info = json.load(f)
    included_ids = set(data_ids_info['ids'])
    
    data = json.load(open(os.path.join(args.dataset_path, 'data.json')))
    
    data = [item for item in data 
            if item.get("task_type") == "Perspective_Taking" 
            and item.get("sub_task_type") == "Allocentric"
            and item["id"] in included_ids]
    
    data = data[args.group_index::args.group]
    
    extraction_cache_path = os.path.join(OUTPUT_DIR, f"extraction_cache_{args.group_index}.json")
    viewpoint_cache_path = os.path.join(OUTPUT_DIR, f"viewpoint_cache_{args.group_index}.json")
    
    extraction_cache = {}
    viewpoint_cache = {}
    
    if os.path.exists(extraction_cache_path):
        with open(extraction_cache_path, "r") as f:
            extraction_cache = json.load(f)
    
    if os.path.exists(viewpoint_cache_path):
        with open(viewpoint_cache_path, "r") as f:
            viewpoint_cache = json.load(f)
    
    results = []
    logs = []
    
    for item in tqdm(data, desc="Processing"):
        with torch.no_grad():
            item_id = item["id"]
            question = item["question"]
            options = item["options"]
            answer_idx = item["answer"]
            answer_letter = chr(65 + answer_idx)
            
            image_path = os.path.join(
                args.dataset_path, 
                item["task_type"], 
                f"{item_id.split('_')[0]}.png"
            )
            image_path, image = resize_image_if_needed(image_path, max_size=args.max_image_size)
            
            axes = categorize_options(options)
            
            if not axes:
                print(f"Warning: No valid axes found for item {item_id}. Skipping.")
                continue
            
            is_multi_axis = len(axes) > 1
            
            axes_info = {}
            
            for axis_name, axis_options in axes.items():
                cache_key = f"{item_id}_{axis_name}"
                
                if cache_key in extraction_cache:
                    ref_obj = extraction_cache[cache_key]["ref_obj"]
                    target_obj = extraction_cache[cache_key]["target_obj"]
                else:
                    ref_obj, target_obj = extract_objects_via_llm(
                        question, options, axis_name, model, processor, device
                    )
                    extraction_cache[cache_key] = {"ref_obj": ref_obj, "target_obj": target_obj}
                    with open(extraction_cache_path, "w") as f:
                        json.dump(extraction_cache, f, indent=2, ensure_ascii=False)
                
                if cache_key in viewpoint_cache:
                    vp_raw = viewpoint_cache[cache_key]["raw"]
                    is_same_vp = viewpoint_cache[cache_key]["is_same"]
                else:
                    vp_raw, is_same_vp = check_same_viewpoint(
                        image, ref_obj, axis_name, model, processor, device
                    )
                    viewpoint_cache[cache_key] = {"raw": vp_raw, "is_same": is_same_vp}
                    with open(viewpoint_cache_path, "w") as f:
                        json.dump(viewpoint_cache, f, indent=2, ensure_ascii=False)
                
                rewrite_q = build_rewrite_q(ref_obj, target_obj, axis_name)
                camera_prompt = build_camera_prompt(rewrite_q, options)
                
                axes_info[axis_name] = {
                    "options": axis_options,
                    "ref_obj": ref_obj,
                    "target_obj": target_obj,
                    "needs_cd": not is_same_vp,
                    "vp_raw": vp_raw,
                    "camera_prompt": camera_prompt
                }
            
            original_prompt = build_original_prompt(question, options)
            
            any_needs_cd = any(info["needs_cd"] for info in axes_info.values())
            
            if not any_needs_cd and not is_multi_axis:
                axis_name = list(axes.keys())[0]
                axis_info = axes_info[axis_name]
                rewrite_q = build_rewrite_q(axis_info["ref_obj"], axis_info["target_obj"], axis_name)
                camera_prompt = build_camera_prompt(rewrite_q, options)
                
                pred_letter = direct_generation_single_axis(
                    image, camera_prompt, model, processor, device, options
                )
            else:
                pred_letter = magnitude_normalized_cd(
                    image=image,
                    original_prompt=original_prompt,
                    axes_info=axes_info,
                    options=options,
                    model=model,
                    processor=processor,
                    device=device,
                    option_token_ids=option_token_ids,
                    axis_alphas=axis_alphas
                )
            
            hit = int(pred_letter == answer_letter)
            
            log_entry = {
                "id": item_id,
                "question": question,
                "options": options,
                "prediction": pred_letter,
                "answer": answer_letter,
                "hit": hit
            }
            
            logs.append(log_entry)
            results.append(hit)
            
            if "_resized" in image_path and os.path.exists(image_path):
                os.remove(image_path)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    result_file = os.path.join(OUTPUT_DIR, f"results_{args.group_index}.json")
    log_file = os.path.join(OUTPUT_DIR, f"logs_{args.group_index}.json")
    
    with open(result_file, "w") as f:
        json.dump({
            "accuracy": sum(results) / len(results) if results else 0,
            "total_samples": len(results)
        }, f, indent=2)
    
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    
    accuracy = sum(results) / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to: {result_file}")
    print(f"Logs saved to: {log_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()