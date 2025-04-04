"""
This script utilizes the Donut model, a VisionEncoderDecoderModel from the Hugging Face Transformers library, 
to perform Optical Character Recognition (OCR) and information extraction from an input image. Specifically, 
it uses a pre-trained model fine-tuned on the CORD dataset for extracting structured data from receipts.

The script performs the following steps:
1. Loads the pre-trained Donut model and processor for the CORD dataset.
2. Prepares the input image by converting it to RGB format and processing it into pixel values.
3. Sets a task-specific prompt for the model to guide the generation process.
4. Generates a sequence of tokens representing the extracted information from the image.
5. Decodes the generated tokens into a human-readable format and converts the output into structured JSON.

The final output is printed as a structured JSON object containing the parsed information from the input image.
"""

from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re

# Use the CORD-finetuned model for receipts
model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
model.eval()

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load and preprocess image
image = Image.open("image.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# Set the task prompt
task_prompt = "<s_cord-v2>"

# Tokenize prompt
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

# Generate output
generated_ids = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]]
)

# Decode output
sequence = processor.batch_decode(generated_ids)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

# Convert to structured JSON
output = processor.token2json(sequence)

print("📄 Parsed Output:\n", output)
