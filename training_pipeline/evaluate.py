
import re
import json
import torch
from tqdm.auto import tqdm
import numpy as np
import yaml
from donut import JSONParseEvaluator
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel


with open("config.yaml", "r") as f:
        config_yaml = yaml.safe_load(f)

# Load the configuration
dataset_hf = config_yaml["DATASET_HF"]
model_hf = config_yaml["HF_MODEL_NAME"]


# Load the dataset and model
processor = DonutProcessor.from_pretrained(model_hf)
model = VisionEncoderDecoderModel.from_pretrained(model_hf)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

def evaluate_model(model, processor, dataset_hf):
    output_list = []
    accs = []

    val_dataset = load_dataset(dataset_hf, split="validation")
    # val_dataset = dataset["validation"]

    for idx, sample in tqdm(enumerate(val_dataset), total=len(val_dataset)):
        # prepare encoder inputs
        pixel_values = processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
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
        print("seq: ", seq)
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
        break


    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    print(scores, f"length : {len(accs)}")

    return output_list, accs, scores

if __name__ == "__main__":
    output_list, accs, scores = evaluate_model(model, processor, dataset_hf)

    # Save the results
    # with open("evaluation_results.json", "w") as f:
    #     json.dump({"outputs": output_list, "accuracies": accs, "scores": scores}, f, indent=4)
    # print("Evaluation completed and results saved to evaluation_results.json")