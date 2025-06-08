
import re
import json
import torch
from tqdm.auto import tqdm
import numpy as np
import yaml
from donut import JSONParseEvaluator
from datasets import load_dataset, load_dataset_builder
from transformers import DonutProcessor, VisionEncoderDecoderModel


with open("config_eval.yaml", "r") as f:
        config_yaml = yaml.safe_load(f)

# Load the configuration
dataset_hf = config_yaml["DATASET_HF"]
model_hf = config_yaml["HF_MODEL_NAME"]
PROMPT_TOKEN = config_yaml["PROMPT_TOKEN"]
evaluation_split = config_yaml["EVALUATION_SPLIT"]
evaluation_result_path = config_yaml["EVALUATION_RESULT_PATH"]


# Load the dataset and model
processor = DonutProcessor.from_pretrained(model_hf)
model = VisionEncoderDecoderModel.from_pretrained(model_hf)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

def evaluate_model(model, processor, dataset_hf):
    output_list = []
    accs = []
    preds_for_f1 = []  
    gts_for_f1 = []    

    # 1. Get the number of examples in that split *without* downloading/processing the data
    builder = load_dataset_builder(dataset_hf)
    num_samples = builder.info.splits[evaluation_split].num_examples

    val_dataset = load_dataset(dataset_hf, split=evaluation_split, streaming=True)
    # val_dataset = dataset["validation"]

    for idx, sample in tqdm(enumerate(val_dataset), num_samples):
        # prepare encoder inputs
        pixel_values = processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        # prepare decoder inputs
        task_prompt = PROMPT_TOKEN
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)

        # autoregressively generate sequence
        outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        # print("seq: ", seq)
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor.token2json(seq)

        ground_truth = json.loads(sample["ground_truth"])
        ground_truth = ground_truth["gt_parse"]
        evaluator = JSONParseEvaluator()
        score = evaluator.cal_acc(seq, ground_truth)

        accs.append(score)
        output_list.append(seq)
        # print("seq: ", seq)
        # break

        # -------- FOR F1 ----------
        preds_for_f1.append(seq)
        gts_for_f1.append(ground_truth)

        output_list.append(seq)

        # if idx > 10:
        #      break

    # ---- aggregate scores ----
    evaluator = JSONParseEvaluator()                      # fresh instance
    field_f1 = evaluator.cal_f1(preds_for_f1, gts_for_f1) # <─ NEW line


    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs), "field_f1": float(field_f1)}
    print(scores, f"length : {len(accs)}")

    return output_list, accs, scores

if __name__ == "__main__":
    output_list, accs, scores = evaluate_model(model, processor, dataset_hf)

    # Save the results
    with open(evaluation_result_path, "w") as f:
        json.dump({"outputs": output_list, "accuracies": accs, "scores": scores}, f, indent=4)
    print(f"Evaluation completed and results saved to {evaluation_result_path}")