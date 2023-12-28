import fire
import random
import torch
# import argparse
from llama_recipes.models.slam_model import slam_model
# config
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import model_config as MODEL_CONFIG
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.pipeline.model_factory import model_factory
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

def main(**kwargs):

	# Update the configuration for the training and sharding process
	train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
	update_config((train_config, fsdp_config, model_config), **kwargs)
	
	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	model.to(device)
	model.eval()

	dataset_config = generate_dataset_config(train_config, kwargs)
	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
	if not train_config.enable_fsdp or rank == 0:
		print(f"--> Test Set Length = {len(dataset_test)}")

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			collate_fn=dataset_test.collator
        )
	

	print("=====================================")
	pred_path = kwargs.get('decode_log') + "_pred"
	gt_path = kwargs.get('decode_log') + "_gt"
	with open(pred_path, "w") as pred, open(gt_path, "w") as gt:
		for step, batch in enumerate(test_dataloader):
			for key in batch.keys():
				batch[key] = batch[key].to(device) if key not in ["keys", "targets"] else batch[key]
			model_outputs = model.generate(**batch)
			output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
			for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
				pred.write(key + "\t" + text.replace("\n", " ") + "\n")
				gt.write(key + "\t" + target + "\n")


if __name__ == "__main__":
	fire.Fire(main)