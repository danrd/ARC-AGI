from tqdm import tqdm
import torch
import json
import os
from transformers.trainer_utils import has_length
from tqdm import tqdm
from transformers.trainer_utils import has_length
from transformers import TrainerCallback

class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, output_dir, tokenizer, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len
        self.output_dir = output_dir
        self.tokenizer = tokenizer
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

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        model.eval()
        predictions = []
        references = []

        for batch in eval_dataloader:
            inputs = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            # Generate predictions
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_length=4048,
                )

            # Decode predictions and references
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_refs = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

        # Save predictions and references to a file
        output_file = os.path.join(self.output_dir, f"predictions_eval_step_{state.global_step}.json")
        with open(output_file, "w") as f:
            json.dump({"predictions": predictions, "references": references}, f, indent=4)

        print(f"Predictions saved to {output_file}")

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
