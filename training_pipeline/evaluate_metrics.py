import re, json, torch, yaml, numpy as np
from tqdm.auto import tqdm
from donut import JSONParseEvaluator
from datasets import load_dataset, load_dataset_builder
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ---------------- load config / model ----------------
with open("config_eval.yaml") as f:
    cfg = yaml.safe_load(f)

dataset_hf       = cfg["DATASET_HF"]
model_hf         = cfg["HF_MODEL_NAME"]
PROMPT_TOKEN     = cfg["PROMPT_TOKEN"]
split            = cfg["EVALUATION_SPLIT"]
result_path      = cfg["EVALUATION_RESULT_PATH"]

processor = DonutProcessor.from_pretrained(model_hf)
model      = VisionEncoderDecoderModel.from_pretrained(model_hf).eval()
device     = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

# ---------------- helpers for streaming metrics ----------------
evaluator = JSONParseEvaluator()

running_ted_sum  = 0.0
count_docs       = 0

tp = fp = fn = 0  # counters for micro-averaged field-F1
mean_ted = 0.0 # nTED accuracy
scores = {} # dict to store results
precision = recall = field_f1 = 0.0 # # F1 score for fields

# ----------------- for inspection -------------------
 
KEEP_EXAMPLES = 20          # how many predictions to keep for inspection
sample_preds  = []          # tiny list, won’t blow up RAM


# --------- dataset size without loading it -------------
num_samples = load_dataset_builder(dataset_hf).info.splits[split].num_examples
ds_stream   = load_dataset(dataset_hf, split=split, streaming=True)

def flatten_dict(d, parent_key="", sep="/"):
    """
    Recursively flattens any mix of dicts & lists.
    Keys become 'parent/child/0/grandchild' strings.
    All leaf values are cast to str for stable equality tests.
    """
    items = {}

    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_dict(v, new_key, sep=sep))

    elif isinstance(d, list):
        for idx, v in enumerate(d):
            new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            items.update(flatten_dict(v, new_key, sep=sep))

    else:  # scalar leaf
        items[parent_key] = str(d)

    return items



# -------------------------------------------------------
for sample in tqdm(ds_stream, total=num_samples, desc="evaluating"):
    try:
        pixel_values = processor(sample["image"].convert("RGB"),
                                return_tensors="pt").pixel_values.to(device)

        decoder_prompt = processor.tokenizer(
            PROMPT_TOKEN, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        out = model.generate(
            pixel_values,
            decoder_input_ids=decoder_prompt,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            num_beams=1,
            early_stopping=True,
            use_cache=True,
            return_dict_in_generate=True,
        )

        pred = processor.batch_decode(out.sequences)[0]
        pred = pred.replace(processor.tokenizer.eos_token, "")\
                .replace(processor.tokenizer.pad_token, "")
        pred = re.sub(r"<.*?>", "", pred, count=1).strip()
        pred = processor.token2json(pred)

        gt   = json.loads(sample["ground_truth"])["gt_parse"]

        # ---------- nTED accuracy ----------
        ted_acc = evaluator.cal_acc(pred, gt)
        running_ted_sum += ted_acc

        # ---------- field-level TP / FP / FN ----------
        p_flat = flatten_dict(pred)
        g_flat = flatten_dict(gt)

        for k, v in p_flat.items():
            if k in g_flat and v == g_flat[k]:
                tp += 1
            elif k in g_flat:
                fp += 1   # wrong value
            else:
                fp += 1   # extra field

        for k, v in g_flat.items():
            if k not in p_flat:
                fn += 1   # missing field
    except Exception as e:
        print(f"Error processing sample: {e}")
        continue

    # -------- keep only a few preds for inspection -----
    if len(sample_preds) < KEEP_EXAMPLES:
        sample_preds.append({"pred": pred, "gt": gt})

    count_docs += 1
# ---------------- aggregate ---------------------------
    mean_ted = running_ted_sum / count_docs

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    field_f1  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    scores = {
        "mean_ted_accuracy": mean_ted,
        "field_f1": field_f1,
        "num_docs": count_docs,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
    }
    # print(scores)

    if count_docs % 100 == 0:
        print(f"Processed {count_docs} documents, mean TED accuracy: {mean_ted:.4f}, field F1: {field_f1:.4f}")

        # -------------- save lite results ---------------------
        with open(result_path, "w") as f:
            json.dump(
                {
                    "scores": scores,
                    "samples": sample_preds,   # only KEEP_EXAMPLES items
                },
                f,
                indent=2,
            )
        print("Saved results to", result_path)
