#!/bin/bash
export PATH=/home/${USER}/.local/bin:${PATH}
accelerate config default

python3 download_dog.py

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="./saved_model"
export CLASS_DIR="./class_folder"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --hub_token=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 \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate=2e-6  \
  --lr_scheduler="constant" \
  --lr_warmup_steps=50  \
  --num_class_images=50 \
  --checkpointing_steps=100 \
  --max_train_steps=400
