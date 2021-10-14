import argparse
import os

import constants
import rama_config
import runner
from run_modes import parallel_run, single_run, utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--mode", metavar="-M", required=True, help="run experiment.")
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="config.yaml",
    help="path to base configuration file.",
)
parser.add_argument(
    "--seeds", metavar="-S", default="[0]", help="list of seeds to run."
)
parser.add_argument("--config_changes", metavar="-CC", default="config_changes.py")


if __name__ == "__main__":

    args = parser.parse_args()

    results_folder = os.path.join(MAIN_FILE_PATH, constants.RESULTS)

    runner_class = runner.Runner
    config_class = rama_config.RamaConfig

    if args.mode == constants.SINGLE:

        single_checkpoint_path = utils.setup_experiment(
            mode="single", results_folder=results_folder, config_path=args.config_path
        )

        single_run.single_run(
            runner_class=runner_class,
            config_class=config_class,
            config_path=args.config_path,
            checkpoint_path=single_checkpoint_path,
            run_methods=["train"],
        )

    elif args.mode == constants.PARALLEL:

        seeds = utils.process_seed_arguments(args.seeds)

        checkpoint_paths = utils.setup_experiment(
            mode="parallel",
            results_folder=results_folder,
            config_path=args.config_path,
            config_changes_path=args.config_changes,
            seeds=seeds,
        )

        parallel_run.parallel_run(
            runner_class=runner_class,
            config_class=config_class,
            config_path=args.config_path,
            checkpoint_paths=checkpoint_paths,
            run_methods=["train"],
            stochastic_packages=["numpy", "torch", "random"],
        )
