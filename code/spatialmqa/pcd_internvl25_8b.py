import os
import json
import argparse
import re

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                 for i in range(1, n + 1)
                 for j in range(1, n + 1)
                 if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    best = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))
    cols, rows = best
    tile_w = image_size * cols
    tile_h = image_size * rows

    resized = image.resize((tile_w, tile_h), Image.BICUBIC)
    tiles = []
    for row in range(rows):
        for col in range(cols):
            box = (col * image_size, row * image_size,
                   (col + 1) * image_size, (row + 1) * image_size)
            tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), Image.BICUBIC))

    return tiles


def load_image_tensor(image: Image.Image, input_size=448, max_num=12):
    tiles = dynamic_preprocess(image, image_size=input_size,
                               use_thumbnail=True, max_num=max_num)
    transform = build_transform(input_size)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCD (InternVL2.5-8B) on SpatialMQA"
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=2.1,
        help="Alpha parameter for left_right axis"
    )
    parser.add_argument(
        "--alpha_fb",
        type=float,
        default=4.0,
        help="Alpha parameter for front_behind axis"
    )
    parser.add_argument(
        "--alpha_ab",
        type=float,
        default=1.0,
        help="Alpha parameter for above_below axis"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/test.jsonl",
        help="Path to the JSONL data file"
    )
    parser.add_argument(
        "--data_images_path",
        type=str,
        default="../../data/data_images.json",
        help="Path to data_images.json file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="../../data/spatialmqa/Dataset/COCO2017/test2017/",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../../outputs/spatialmqa",
        help="Root directory for outputs"
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=12,
        help="Max number of dynamic tiles per image (default: 12)"
    )
    return parser.parse_args()


def load_data(jsonl_path, images_json_path):
    with open(images_json_path, 'r') as f:
        images_data = json.load(f)
    included_images = set(images_data['images'])
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if item['image'] in included_images:
                data.append(item)
    
    return data


def categorize_options(options):
    axes = {}
    
    left_right_terms = ["left of", "right of"]
    front_behind_terms = ["in front of", "behind"]
    above_below_terms = ["on/above", "above", "on", "below"]
    
    lr = [opt for opt in options if opt in left_right_terms]
    fb = [opt for opt in options if opt in front_behind_terms]
    ab = [opt for opt in options if opt in above_below_terms]
    
    if lr:
        axes["left_right"] = lr
    if fb:
        axes["front_behind"] = fb
    if ab:
        axes["above_below"] = ab
    
    return axes


def get_question_type(options):
    return len(options)


EXTRACTION_PROMPTS = {
    "left_right": """\
Extract the "reference object" and "target object" from a spatial reasoning question.
- reference object: the object whose position and facing direction defines the viewpoint
- target object: the object whose left/right position is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: If you were a woman walking on the beach, on which side of you would the sunbed be?
A: {"ref_obj": "woman walking on the beach", "target_obj": "sunbed"}

Q: If you are the athlete in the image, where is the cat located relative to you?
A: {"ref_obj": "athlete", "target_obj": "cat"}

Q: If you were the driver of the bus in the image, where would the gray car be located relative to you?
A: {"ref_obj": "driver of the bus", "target_obj": "gray car"}

Q: {question}
A:""",
    
    "front_behind": """\
Extract the "reference object" and "target object" from a spatial reasoning question.
- reference object: the object whose position and facing direction defines the viewpoint
- target object: the object whose front/behind position is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: Where is the cup located relative to the microwave?
A: {"ref_obj": "microwave", "target_obj": "cup"}

Q: If you were the person wearing the black hat and holding the bag in the image, where would the switch be located relative to you?
A: {"ref_obj": "person wearing the black hat and holding the bag", "target_obj": "switch"}

Q: If you were the man in the image, where would the kite be located relative to you?
A: {"ref_obj": "man", "target_obj": "kite"}

Q: {question}
A:""",
    
    "above_below": """\
Extract the "reference object" and "target object" from a spatial reasoning question.
- reference object: the object whose upright orientation defines the viewpoint
- target object: the object whose above/below position is being asked about

Return your answer strictly in this JSON format (no extra text):
{"ref_obj": "<reference object>", "target_obj": "<target object>"}

Examples:

Q: Where is the mirror located relative to the computer?
A: {"ref_obj": "computer", "target_obj": "mirror"}

Q: Where is the green sign located relative to the electronic board?
A: {"ref_obj": "electronic board", "target_obj": "green sign"}

Q: {question}
A:"""
}


@torch.no_grad()
def extract_objects_via_llm(question, axis_name, model, tokenizer, device):
    prompt_template = EXTRACTION_PROMPTS.get(axis_name, EXTRACTION_PROMPTS["left_right"])
    prompt = prompt_template.replace("{question}", question.strip())

    generation_config = dict(max_new_tokens=64, do_sample=False)

    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=None,
        question=prompt,
        generation_config=generation_config,
        history=None,
        return_history=False,
    )
    generated_text = response.strip()

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


def build_viewpoint_check_prompt(ref_obj, axis_name):
    if axis_name == "left_right":
        return (
            f"Consider the real-world 3D locations and orientations of the objects.\n"
            f"Question:\n"
            f"In image, are you viewing the {ref_obj}'s back side?\n"
            f"Answer only yes or no."
        )
    elif axis_name == "front_behind":
        return (
            f"Question: Consider the real-world 3D locations and orientations of the objects."
            f"In image, are you looking at the {ref_obj} from behind?"
            f"Answer only yes or no."
        )
    elif axis_name == "above_below":
        return (
            f"Question: Consider the real-world 3D positions and orientations of the objects. "
            f"In the image, is the {ref_obj} upside down (head at bottom, feet at top)? "
            f"Answer only yes or no."
        )
    else:
        return ""


@torch.no_grad()
def generate_answer(pixel_values, prompt, model, tokenizer, device, max_tiles):
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=torch.float16)
        num_patches_list = [pixel_values.shape[0]]
    else:
        num_patches_list = None

    generation_config = dict(max_new_tokens=20, do_sample=False)

    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    return response.strip()


@torch.no_grad()
def check_same_viewpoint(pixel_values, ref_obj, axis_name, model, tokenizer, device, max_tiles):
    prompt = build_viewpoint_check_prompt(ref_obj, axis_name)
    raw = generate_answer(pixel_values, prompt, model, tokenizer, device, max_tiles)
    
    if axis_name == "left_right":
        is_same = raw.lower().startswith("yes")
    else:
        is_same = raw.lower().startswith("no")
    
    return raw, is_same


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


def build_original_prompt(question, options):
    labels = ["A", "B", "C", "D", "E", "F"][:len(options)]
    options_text = "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options)])
    
    question_body = extract_question_body(question)
    
    return f"""Question: {question_body}

Options:
{options_text}

Return only the correct option letter.
"""


def build_camera_prompt(question, ref_obj, target_obj, options, axis_name):
    labels = ["A", "B", "C", "D", "E", "F"][:len(options)]
    options_text = "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options)])
    
    if axis_name == "left_right":
        rewrite_q = f"Is the {target_obj} on the left or right of the {ref_obj}?"
    elif axis_name == "front_behind":
        rewrite_q = f"is the {target_obj} in front of the {ref_obj} or behind the {ref_obj}?"
    elif axis_name == "above_below":
        rewrite_q = f"From the camera view, is the {target_obj} above the {ref_obj} or below the {ref_obj}?"
    else:
        rewrite_q = f"From the camera view, is the {target_obj} to the left or right of the {ref_obj}?"
    
    return f"""Question: From the camera view, {rewrite_q}

Options:
{options_text}

Return only the correct option letter.
"""


def build_direct_generation_prompt(question, ref_obj, target_obj, options, axis_name):
    return build_camera_prompt(question, ref_obj, target_obj, options, axis_name)


def get_candidate_token_ids(label, tokenizer):
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


def _ensure_img_context_token_id(model, tokenizer):
    if getattr(model, "img_context_token_id", None) is None:
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


def _build_internvl_inputs(pixel_values, prompt, model, tokenizer, device):
    _ensure_img_context_token_id(model, tokenizer)

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN   = "<img>"
    IMG_END_TOKEN     = "</img>"

    img_context_token_id = model.img_context_token_id
    num_image_token = model.num_image_token

    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=torch.float16)
        num_tiles    = pixel_values.shape[0]
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * num_image_token * num_tiles
            + IMG_END_TOKEN
        )
    else:
        image_tokens = ""

    system_message = "You are an AI assistant whose name is InternLM."
    conversation = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    encoded = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)

    ctx_enc_id = tokenizer.encode(IMG_CONTEXT_TOKEN, add_special_tokens=False)
    if len(ctx_enc_id) == 1 and ctx_enc_id[0] != img_context_token_id:
        input_ids = input_ids.clone()
        input_ids[input_ids == ctx_enc_id[0]] = img_context_token_id

    attention_mask = torch.ones_like(input_ids)

    inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
    if pixel_values is not None:
        inputs["pixel_values"] = pixel_values

    return inputs


@torch.no_grad()
def get_next_token_logprobs(pixel_values, prompt, model, tokenizer, device, max_tiles):
    inputs = _build_internvl_inputs(pixel_values, prompt, model, tokenizer, device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    logits = outputs.scores[0][0]
    logp   = F.log_softmax(logits, dim=-1)
    return logp


def magnitude_normalized_cd(
    pixel_values, 
    original_prompt,
    axis_to_camera_prompt,
    axes_info,
    options,
    model,
    tokenizer,
    device,
    option_token_ids,
    axis_alphas,
    max_tiles
):
    labels = ["A", "B", "C", "D", "E", "F"][:len(options)]
    
    orig_logp = get_next_token_logprobs(pixel_values, original_prompt, model, tokenizer, device, max_tiles)
    
    orig_scores = {}
    for i, opt in enumerate(options):
        label = labels[i]
        token_ids = option_token_ids[label]
        orig_scores[i] = max(orig_logp[tid].item() for tid in token_ids)
    
    final_scores = orig_scores.copy()
    
    for axis_name, info in axes_info.items():
        if not info["needs_cd"]:
            continue
        
        camera_prompt = axis_to_camera_prompt[axis_name]
        cam_logp = get_next_token_logprobs(pixel_values, camera_prompt, model, tokenizer, device, max_tiles)
        
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
    
    return pred_idx


@torch.no_grad()
def direct_generation(pixel_values, prompt, model, tokenizer, device, num_options, max_tiles):
    generated = generate_answer(pixel_values, prompt, model, tokenizer, device, max_tiles)
    
    labels = ["A", "B", "C", "D", "E", "F"][:num_options]
    
    generated_upper = generated.upper()
    for i, label in enumerate(labels):
        if label in generated_upper:
            return i
    
    match = re.search(r'[A-F]', generated_upper)
    if match:
        letter = match.group()
        if letter in labels:
            return labels.index(letter)
    
    print(f"Warning: Could not parse letter from generated text: '{generated}'")
    return 0


def process_sample(
    sample,
    sample_idx,
    model,
    tokenizer,
    device,
    image_dir,
    option_token_ids,
    extraction_cache,
    viewpoint_cache,
    axis_alphas,
    max_tiles
):
    qid = f"sample_{sample_idx}"
    question = sample["question"]
    options = sample["options"]
    answer_text = sample["answer"]
    answer_idx = options.index(answer_text)
    image_file = sample["image"]
    
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    pixel_values = load_image_tensor(image, max_num=max_tiles)
    
    axes = categorize_options(options)
    question_type = get_question_type(options)
    
    axes_info = {}
    axis_to_camera_prompt = {}
    
    for axis_name, axis_options in axes.items():
        cache_key = f"{qid}_{axis_name}"
        
        if cache_key in extraction_cache:
            ref_obj = extraction_cache[cache_key]["ref_obj"]
            target_obj = extraction_cache[cache_key]["target_obj"]
        else:
            ref_obj, target_obj = extract_objects_via_llm(
                question, axis_name, model, tokenizer, device
            )
            extraction_cache[cache_key] = {"ref_obj": ref_obj, "target_obj": target_obj}
        
        if cache_key in viewpoint_cache:
            vp_raw = viewpoint_cache[cache_key]["raw"]
            is_same_vp = viewpoint_cache[cache_key]["is_same"]
        else:
            vp_raw, is_same_vp = check_same_viewpoint(
                pixel_values, ref_obj, axis_name, model, tokenizer, device, max_tiles
            )
            viewpoint_cache[cache_key] = {"raw": vp_raw, "is_same": is_same_vp}
        
        axes_info[axis_name] = {
            "options": axis_options,
            "ref_obj": ref_obj,
            "target_obj": target_obj,
            "needs_cd": not is_same_vp,
            "vp_raw": vp_raw
        }
        
        camera_prompt = build_camera_prompt(
            question, ref_obj, target_obj, options, axis_name
        )
        axis_to_camera_prompt[axis_name] = camera_prompt
    
    first_axis = list(axes.keys())[0] if axes else None
    original_prompt = build_original_prompt(question, options)
    
    if len(axes) == 1:
        first_axis_info = axes_info[first_axis]
        direct_prompt = build_direct_generation_prompt(
            question, 
            first_axis_info["ref_obj"], 
            first_axis_info["target_obj"], 
            options, 
            first_axis
        )
    else:
        direct_prompt = original_prompt
    
    any_needs_cd = any(info["needs_cd"] for info in axes_info.values())
    
    if not any_needs_cd and question_type == 2:
        pred_idx = direct_generation(pixel_values, original_prompt, model, tokenizer, device, len(options), max_tiles)
    else:
        pred_idx = magnitude_normalized_cd(
            pixel_values=pixel_values,
            original_prompt=original_prompt,
            axis_to_camera_prompt=axis_to_camera_prompt,
            axes_info=axes_info,
            options=options,
            model=model,
            tokenizer=tokenizer,
            device=device,
            option_token_ids=option_token_ids,
            axis_alphas=axis_alphas,
            max_tiles=max_tiles
        )
    
    if pred_idx < 0 or pred_idx >= len(options):
        print(f"Warning: Invalid pred_idx {pred_idx} for {len(options)} options. Defaulting to 0.")
        pred_idx = 0
    
    correct = int(pred_idx == answer_idx)
    
    return {
        "qid": qid,
        "question": question,
        "options": options,
        "prediction": options[pred_idx],
        "answer": options[answer_idx],
        "hit": correct
    }


def main():
    args = parse_args()
    
    axis_alphas = {
        "left_right": args.alpha_lr,
        "front_behind": args.alpha_fb,
        "above_below": args.alpha_ab
    }
    
    MODEL_NAME = "OpenGVLab/InternVL2_5-8B"
    
    alpha_desc = f"lr{args.alpha_lr}_fb{args.alpha_fb}_ab{args.alpha_ab}"
    alpha_desc = alpha_desc.replace(".", "p")
    model_name_safe = f"PCD_InternVL2_5-8B_{alpha_desc}"
    
    OUTPUT_DIR = os.path.join(args.output_root, model_name_safe)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {device}")
    print(f"Max tiles: {args.max_tiles}")
    print(f"{'='*60}\n")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
    )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    _ensure_img_context_token_id(model, tokenizer)
    print("Model loaded successfully!\n")
    
    option_token_ids = {}
    for label in ["A", "B", "C", "D", "E", "F"]:
        option_token_ids[label] = get_candidate_token_ids(label, tokenizer)
    
    data = load_data(args.data_path, args.data_images_path)
    
    extraction_cache_path = os.path.join(OUTPUT_DIR, "extraction_cache.json")
    viewpoint_cache_path = os.path.join(OUTPUT_DIR, "viewpoint_cache.json")
    
    extraction_cache = {}
    viewpoint_cache = {}
    
    if os.path.exists(extraction_cache_path):
        with open(extraction_cache_path, "r") as f:
            extraction_cache = json.load(f)
    
    if os.path.exists(viewpoint_cache_path):
        with open(viewpoint_cache_path, "r") as f:
            viewpoint_cache = json.load(f)
    
    results = []
    
    for idx, sample in enumerate(tqdm(data, desc="Processing")):
        result = process_sample(
            sample=sample,
            sample_idx=idx,
            model=model,
            tokenizer=tokenizer,
            device=device,
            image_dir=args.image_dir,
            option_token_ids=option_token_ids,
            extraction_cache=extraction_cache,
            viewpoint_cache=viewpoint_cache,
            axis_alphas=axis_alphas,
            max_tiles=args.max_tiles
        )
        results.append(result)
        
        if len(results) % 100 == 0:
            with open(extraction_cache_path, "w") as f:
                json.dump(extraction_cache, f, indent=2, ensure_ascii=False)
            with open(viewpoint_cache_path, "w") as f:
                json.dump(viewpoint_cache, f, indent=2, ensure_ascii=False)
    
    with open(extraction_cache_path, "w") as f:
        json.dump(extraction_cache, f, indent=2, ensure_ascii=False)
    with open(viewpoint_cache_path, "w") as f:
        json.dump(viewpoint_cache, f, indent=2, ensure_ascii=False)
    
    logs_path = os.path.join(OUTPUT_DIR, "logs.json")
    with open(logs_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    total = len(results)
    correct = sum(r["hit"] for r in results)
    accuracy = correct / total if total > 0 else 0
    
    results_summary = {
        "accuracy": accuracy,
        "total_samples": total
    }
    
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to: {results_path}")
    print(f"Logs saved to: {logs_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()