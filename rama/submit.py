import submitit
from rama import rama_config, runner
from run_modes import single_run

runner_class = runner.Runner
config_class = rama_config.RamaConfig

executor = submitit.AutoExecutor(folder="log_exp")

executor.update_parameters(
    timeout_min=20,
    mem_gb=30,
    gpus_per_node=1,
    cpus_per_task=8,
    slurm_array_parallelism=256,
    slurm_partition="gpu",
)

jobs = []
with executor.batch():
    for seed in range(1):
        job = executor.submit(
            single_run.single_run,
            runner_class=runner_class,
            config_class=config_class,
            config_path="config.yaml",
            checkpoint_path="log_exp",
            run_methods=["train", "post_process"],
        )
