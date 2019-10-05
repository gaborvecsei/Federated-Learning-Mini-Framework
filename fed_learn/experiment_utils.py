import shutil
from pathlib import Path

from swiss_army_tensorboard import tfboard_loggers

import fed_learn


class Experiment:
    def __init__(self, experiment_folder_path: Path, overwrite_if_exists: bool = False):
        self.experiment_folder_path = experiment_folder_path

        if self.experiment_folder_path.is_dir():
            if overwrite_if_exists:
                shutil.rmtree(str(self.experiment_folder_path))
            else:
                raise Exception("Experiment already exists")

        self.experiment_folder_path.mkdir(parents=True, exist_ok=False)

        self.args_json_path = self.experiment_folder_path / "args.json"

        self.train_hist_path = self.experiment_folder_path / "fed_learn_global_test_results.json"
        self.global_weight_path = self.experiment_folder_path / "global_weights.h5"

    def serialize_args(self, args):
        fed_learn.save_args_as_json(args, self.args_json_path)
        tfboard_loggers.TFBoardTextLogger(self.experiment_folder_path).log_markdown("args", "```\n{0}\n```".format(
            fed_learn.args_as_json(args)), -1)

    def create_scalar_logger(self) -> tfboard_loggers.TFBoardScalarLogger:
        tf_scalar_logger = tfboard_loggers.TFBoardScalarLogger(self.experiment_folder_path)
        return tf_scalar_logger
