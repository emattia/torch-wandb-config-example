from metaflow import (
    FlowSpec,
    step,
    resources,
    kubernetes,
    Config,
    IncludeFile,
    checkpoint,
    model,
    current,
    secrets,
    card
)
from metaflow.profilers import gpu_profile
import os
import yaml


class BertTrainingSingleNodeMultiGPU(FlowSpec):

    config = Config("config", default="conf/metaflow_config.json")
    train_config = IncludeFile(
        name="train_conf",
        required=True,
        default="conf/pytorch_train_config.yaml"
    )
    eval_config = IncludeFile(
        name="eval_conf",
        required=True,
        default="conf/pytorch_eval_config.yaml"
    )

    @step
    def start(self):
        # TODO: Add preliminary "fail fast" checks,
        # where you can look at a config and run
        # assertions and avoid spinning up GPUs
        # if something is off.
        self.next(self.train)

    @gpu_profile(interval=5)
    @model
    @checkpoint
    # @secrets(sources=[...]) ### TODO: On Outerbounds, configure your wandb secret via resource integration.
    @kubernetes(**config.train_resources)
    @step
    def train(self):
        from metaflow import TorchrunSingleNodeMultiGPU

        os.makedirs('conf', exist_ok=True)
        with open("conf/pytorch_train_config.yaml", "w") as f:
            f.write(self.train_config)

        # validate config is saved correctyl, and load for use in Metaflow step code.
        with open("conf/pytorch_train_config.yaml", "r") as f:
            torch_cfg = yaml.safe_load(f)

        executor = TorchrunSingleNodeMultiGPU()
        executor.run(
            entrypoint="train.py",
            nproc_per_node=self.config.train_resources.gpu,
            entrypoint_args=[
                # NOTE: this arg can be dict or list.
                # When train script uses Hydra configs, use list syntax as follows.
                "metaflow.use_metaflow=true",
                "metaflow.checkpoint_in_remote_datastore=true",
                # "wandb.use_wandb=true",
                # "wandb.project=bert-multi-gpu"
            ],
        )
        self.bert_model = current.model.save(
            torch_cfg["metaflow"]["final_model_path"],
            label="%s_%s" % (current.task_id, torch_cfg["model"]["model_name"]),
            metadata={
                "epochs": torch_cfg["training"]["epochs"],
                "batch-size": torch_cfg["training"]["batch_size"],
                "learning-rate": torch_cfg["training"]["learning_rate"],
                "model-name": torch_cfg["model"]["model_name"],
            },
        )
        self.next(self.eval)

    @gpu_profile(interval=5)
    @card
    @model(load="bert_model")
    # @secrets(sources=[...]) ### TODO: On Outerbounds, configure your wandb secret via resource integration.
    @kubernetes(**config.eval_resources)
    @step
    def eval(self):
        from metaflow import TorchrunSingleNodeMultiGPU

        os.makedirs('conf', exist_ok=True)
        with open("conf/pytorch_eval_config.yaml", "w") as f:
            f.write(self.eval_config)

        final_model_path = os.path.join(
            current.model.loaded["bert_model"],
            "final_model.pth",
        )
        eval_results_filepath = "evaluation_results.json"

        executor = TorchrunSingleNodeMultiGPU()
        executor.run(
            entrypoint="eval.py",
            nproc_per_node=self.config.eval_resources.gpu,
            entrypoint_args=[
                # NOTE: this arg can be dict or list.
                # When train script uses Hydra configs, use list syntax as follows.
                "metaflow.use_metaflow=true",
                f"model_path={final_model_path}",
                f"output_file={eval_results_filepath}",
                # "wandb.use_wandb=true",
                # "wandb.project=bert-multi-gpu"
            ],
        )
        self.eval_results = load_evaluation_results(eval_results_filepath)
        self.next(self.end)

    @step
    def end(self):
        if hasattr(self, "eval_results") and "error" not in self.eval_results:
            print(f"Evaluation complete!")
            print(f"Final accuracy: {self.eval_results['accuracy']:.2f}%")
        else:
            print("Evaluation failed or produced invalid results.")
        pass

def load_evaluation_results(filepath):
    import json

    with open(filepath, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    BertTrainingSingleNodeMultiGPU()
