import argparse

from experiment import Experiment

def _parse_args(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Load a dataset and parameters to the retrieval"
    )
    parser.add_argument(
        "dataset_name",
        #choices=["on", "off"],
        default="nfcorpus",
        help="Name of the dataset to load",
    )

    args = parser.parse_args(argv)
    return args

def main(argv: list[str] | None = None) -> None:
    config = _parse_args(argv)
    experiment = Experiment(config)
    experiment.run()
    return

if __name__ == "__main__":
    main()