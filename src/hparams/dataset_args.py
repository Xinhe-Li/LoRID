import argparse


# dataset args
dataset_parser = argparse.ArgumentParser(
    description="Parser For Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
dataset_parser.add_argument(
    "--raw_dataset_path",
    dest="raw_dataset_path",
    default="",
    type=str,
    help="Path to the dataset without knowledge",
)
dataset_parser.add_argument(
    "--dataset_path",
    dest="dataset_path",
    default="",
    type=str,
    help="Path to the dataset with knowledge",
)
dataset_args = dataset_parser.parse_args()
