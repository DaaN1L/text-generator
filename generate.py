import argparse
from pathlib import Path

from model import TextGenerator


def file(path: str):
    p = Path(path).resolve()

    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"readable_path:{path} is not a valid file path.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=file,
        help="the path to the file from which the model is loaded (required)"
    )
    parser.add_argument(
        "--prefix",
        nargs="*",
        help="beginning of a sentence. Can be one or more words. "
             "If not specified, select the initial word randomly "
             "from all the words (optional)"
    )
    parser.add_argument(
        "--length",
        required=True,
        type=int,
        help="length of the generated sequence (required)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to initialize the random number generator"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    text_generator = TextGenerator()
    prefix = args.prefix if args.prefix is None else " ".join(args.prefix)
    generated_text = text_generator.generate(
        model=args.model,
        length=args.length,
        text_beginning=prefix,
        seed=args.seed
    )
    print(generated_text)


if __name__ == '__main__':
    main()
