"""
Inference pipeline: invoice image -> preprocessing -> trained Donut model
-> raw token sequence -> parsed -> normalized -> schema-validated -> JSON.

Usage:
    python infer.py --checkpoint /path/to/invoice_donut_final --image path/to/invoice.jpg
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import schema

def load_model(checkpoint_dir):
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    processor = DonutProcessor.from_pretrained(checkpoint_dir)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def run_inference(image_path, processor, model, device, max_length=768):
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = torch.tensor(
        [[processor.tokenizer.convert_tokens_to_ids([schema.TASK_START_TOKEN])[0]]]
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            return_dict_in_generate=True,
        )

    raw_text = processor.tokenizer.batch_decode(outputs.sequences)[0]
    parsed = schema.from_donut_output(raw_text)     # never raises, degrades to None fields
    normalized = schema.normalize(parsed)
    is_valid, problems = schema.validate_schema(normalized)

    return {
        "structured_json": normalized,
        "is_valid": is_valid,
        "validation_problems": problems,
        "raw_model_output": raw_text,   # kept for debugging, per the assessment's pipeline spec
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    processor, model, device = load_model(args.checkpoint)
    result = run_inference(args.image, processor, model, device)
    print(json.dumps(result["structured_json"], indent=2))
    if not result["is_valid"]:
        print("\n[schema warnings]", result["validation_problems"], file=sys.stderr)
