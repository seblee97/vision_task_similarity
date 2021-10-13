import os

import rama_config
import runner
from run_modes import single_run, utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
RESULTS = "results"
CONFIG_PATH = "config.yaml"

results_folder = os.path.join(MAIN_FILE_PATH, RESULTS)

runner_class = runner.Runner
config_class = rama_config.RamaConfig

single_checkpoint_path = utils.setup_experiment(
    mode="single", results_folder=results_folder, config_path=CONFIG_PATH
)

single_run.single_run(
    runner_class=runner_class,
    config_class=config_class,
    config_path=CONFIG_PATH,
    checkpoint_path=single_checkpoint_path,
    run_methods=["train"],
)
