import argparse

from models import unsloth_model, generation_pipe
from train import train
from inference import inference


def parse_args():
    parser = argparse.ArgumentParser("Product Comparision & Recommendation")

    # training arguments
    parser.add_argument("--train", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dataset", default="Dohyeon1/Product-Comparison-Recommendataion-RAG", type=str)

    # inference arguments
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--model_id", default="Not Implemnted", type=str)

    # lora arguments
    parser.add_argument("--r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    # util arguents
    parser.add_argument("--output_dir", default="./results", type=str)

    args = parser.parse_args()
    return args


def main(args):
    if args.train:
        model, tokenizer = unsloth_model(args)
        train(model, tokenizer, args)
    else:
        pipe = generation_pipe(args)
        response = inference(pipe, args.query)
        print(response)


if __name__ == "__main__":
    args = parse_args()
    main(args)