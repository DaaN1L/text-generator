import argparse
from pathlib import Path

from model import TextGenerator


def directory(path: str):
    p = Path(path).resolve()

    if p.is_dir():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"readable_path:{path} is not a valid dir path.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="path to the .pkl file where the model will be saved (required)"
    )
    parser.add_argument(
        "--input-dir",
        type=directory,
        help="path to the directory where the collection of documents is "
             "located. If this argument is not specified, the texts are "
             "entered from stdin (optional)"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="ngram order"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    text_generator = TextGenerator()
    text_generator.fit(
        input_dir=args.input_dir,
        model=args.model,
        order=args.order
    )


if __name__ == '__main__':
    main()
