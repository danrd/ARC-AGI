import numpy as np
import wandb
from tqdm import tqdm
import torch
import json
import os
from transformers.trainer_utils import has_length
from transformers import TrainerCallback
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn
from llm.utils import lev_sim

class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, output_dir, tokenizer, eval_dataloader, max_str_len:int = 2000):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.eval_dataloader = eval_dataloader
        self.max_str_len = max_str_len
        os.makedirs(output_dir, exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    def on_epoch_end(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        file_name = args.output_dir + f'predictions_ep_{state.epoch}.json'
        mean_sim, accuracy = evaluate_model(model, self.tokenizer, self.eval_dataloader, output_file=file_name)
        wandb.log({"mean_sim": mean_sim, "accuracy": accuracy})

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            self.training_bar.write(str(shallow_logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None

class PLProgressCallback(Callback):
    """
    A PyTorch Lightning callback that saves predictions incrementally to avoid memory issues.
    """

    def __init__(self, output_dir, tokenizer, eval_dataloader, max_str_len=2000):
        super().__init__()
        self.max_str_len = max_str_len
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.eval_dataloader = eval_dataloader
        os.makedirs(output_dir, exist_ok=True)

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #     self._process_batch(trainer, pl_module, batch, batch_idx, "validation")

    def on_train_epoch_end(self, trainer, pl_module):
        file_name = trainer.default_root + f'predictions_ep_{trainer.current_epoch}.json'
        mean_sim, accuracy = evaluate_model(pl_module.model, self.tokenizer, self.eval_dataloader, output_file=file_name)
        wandb.log({"mean_sim": mean_sim, "accuracy": accuracy})

def evaluate_model(model, tokenizer, eval_dataloader, max_new_tokens=1025, output_file='data/predictions.json'):
        model.eval()
        predictions = []
        references = []

        for batch in eval_dataloader:
            inputs = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Generate predictions
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            inputs = torch.where(inputs==-100, tokenizer.pad_token_id, inputs)
            outputs = torch.where(outputs==-100, tokenizer.pad_token_id, outputs)
            labels = torch.where(batch["labels"]==-100, tokenizer.pad_token_id, batch["labels"])
            
            # Decode predictions and references
            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Remove input text from predictions (if necessary)
            for i in range(len(decoded_preds)):
            # Remove the input text from the prediction
                decoded_preds[i] = decoded_preds[i][len(decoded_inputs[i]):]
            
            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

        # Save predictions and references to a file
        with open(output_file, "w") as f:
            json.dump({"predictions": predictions, "references": references}, f, indent=4)

        print(f"Predictions saved to {output_file}")

        similarities = []
        trues_list = []
        for i in range(len(predictions)):
            pred = predictions[i]
            ref = references[i]
            sim = lev_sim(pred, ref)
            similarities.append(sim)
            if sim==1:
                trues_list.append(1)
            else:
                trues_list.append(0)
        mean_sim = np.mean(np.array(similarities))
        accuracy = sum(trues_list) / len(trues_list)
        return mean_sim, accuracy 
