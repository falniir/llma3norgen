import os
import ludwig.visualize
import yaml
import logging
from datasets import load_dataset
from ludwig.api import LudwigModel
import pandas as pd
from dotenv import dotenv_values
import ludwig

env_config = dotenv_values(".env")


# Load the dataset
dataset = load_dataset("NbAiLab/norwegian-alpaca")


dataset_df = dataset['train'].to_pandas()

config_str = """
model_type: llm
base_model: meta-llama/Meta-Llama-3-8B
quantization:
  bits: 4
adapter:
  type: lora
prompt:
  template: |
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:

input_features:
  - name: instruction
    type: text
    preprocessing:
      max_sequence_length: 256  # Adjusted to allow longer texts

output_features:
  - name: output
    type: text
    preprocessing:
      max_sequence_length: 256

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1  # Adjust as needed based on your dataset size and training needs
"""
config = yaml.safe_load(config_str)



# Initialize and train the model
model = LudwigModel(config=config, logging_level=logging.INFO)

# Now train the model using this DataFrame
results = model.train(dataset_df)  # Ensure the correct subset of the dataset is used

# Save model
model.save("model")


