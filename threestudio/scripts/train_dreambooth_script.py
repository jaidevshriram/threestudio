import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--instance_dir', \
                required = True, \
                type = str, \
                help = 'Folder containing instance images' \
              )

parser.add_argument('--output_dir', \
                required = False, \
                type = str, \
                default = './saved_model', \
                help = 'Folder to save the model' \
              )

parser.add_argument('--class_dir', \
                required = False, \
                type = str, \
                default = './class_folder', \
                help = 'Folder to store class images' \
              )

parser.add_argument('--instance_prompt', \
                required = True, \
                type = str, \
                help = 'example: A photo of sks dog' \
              )

parser.add_argument('--class_prompt', \
                required = True, \
                type = str, \
                help = 'example: A photo of a dog' \
              )

parser.add_argument('--resolution', type = int, default = 512)
parser.add_argument('--train_batch_size', type = int, default = 1)
parser.add_argument('--gradient_accumulation_steps', type = int, default = 1)
parser.add_argument('--lr', type = float, default = 2e-6)
parser.add_argument('--lr_warmup_steps', type = int, default = 50)
parser.add_argument('--num_class_images', type = int, default = 100)
parser.add_argument('--checkpointing_steps', type = int, default = 50)
parser.add_argument('--max_train_steps', type = int, default = 800)

args = parser.parse_args()

os.environ['PATH'] = '/home/{}/.local/bin'.format(os.environ.get("USERNAME")) + ':' + os.environ['PATH']

os.system('accelerate config default')

MODEL_NAME="CompVis/stable-diffusion-v1-4"
INSTANCE_DIR=args.instance_dir
OUTPUT_DIR=args.output_dir
CLASS_DIR=args.class_dir
INSTANCE_PROMPT = args.instance_prompt
CLASS_PROMPT = args.class_prompt

cmd = 'accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path={}  \
  --instance_data_dir={} \
  --class_data_dir=${} \
  --hub_token=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 \
  --output_dir={} \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt={} \
  --class_prompt={} \
  --resolution={} \
  --train_batch_size={} \
  --gradient_accumulation_steps={} \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate={}  \
  --lr_scheduler="constant" \
  --lr_warmup_steps={}  \
  --num_class_images={} \
  --checkpointing_steps={} \
  --max_train_steps={}'.format(MODEL_NAME, INSTANCE_DIR, CLASS_DIR, OUTPUT_DIR, INSTANCE_PROMPT, CLASS_PROMPT, args.resolution, args.train_batch_size, args.gradient_accumulation_steps, args.lr, args.lr_warmup_steps, args.num_class_images, args.checkpointing_steps, args.max_train_steps)

os.system(cmd)

