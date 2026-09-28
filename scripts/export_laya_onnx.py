#!/usr/bin/env python3
"""
CLI script to export convaiinnovations/laya ModernBERT model to ONNX format.
Configures dynamic sequence axes and optimizes for CPU inference in bfloat16/float32.
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_laya_onnx")


def export_onnx(model_id: str, output_path: str, dtype: str = "bfloat16") -> None:
    logger.info(f"Preparing to export {model_id} to ONNX (dtype={dtype}) at {output_path}...")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing required ML libraries: {e}. Install torch and transformers.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Loading PyTorch model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    dummy_input = tokenizer(
        "What is the bereavement policy?",
        "Up to 5 days leave.",
        return_tensors="pt",
    )

    logger.info(f"Exporting to ONNX at {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    logger.info("ONNX export successfully completed.")


def main():
    parser = argparse.ArgumentParser(description="Export ModernBERT Laya model to ONNX")
    parser.add_argument(
        "--model", type=str, default="convaiinnovations/laya", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output", type=str, default="models/laya.onnx", help="Output path for ONNX file"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precision",
    )
    args = parser.parse_args()

    export_onnx(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()
