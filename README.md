# DreamBoothFusion - Personalized Driven Text to 3D Generation (using threestudio) [CSE252D]

DreamBoothFusion is an open-source implementation of [Dreambooth3D](https://arxiv.org/abs/2303.13508#:~:text=We%20present%20DreamBooth3D%2C%20an%20approach,%2D3D%20generation%20(DreamFusion).) which was introduced for personalized Text-to-3D generation where given the framework uses only 5-6 images of a specific subject to generate subject specific 3D viewpoints in various contexts as provided by the text prompt. In this project we have implemented this approach using the threestudio framework. 
threestudio is a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.


## Installation

The following steps have been tested on Ubuntu20.04.

- You must have a NVIDIA graphics card with at least 6GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.

We have created a custom docker image listed below. Please note that the code is not compatible with GPU Versions lower than GeForce GTX 2080 Ti. 
```sh
launch-scipy-ml.sh -i ghcr.io/shreyasumbetla/custom-docker:nightly -g 1 -m 16 -c 8 -s -v 2080ti
```
You can run the threestudio_setup.sh file which will download this repository and setup the python environment. Then activate the python env using
```sh
./threestudio_setup.sh
. venv/bin/activate
```
We use the StableDiffusion T2I model in the backbone and use the Dreambooth and Dreamfusion model for our approach. The Dreambooth model is taken from  [huggingface](https://huggingface.co/docs/diffusers/main/en/training/dreambooth) while Dreamfusion is implemented within threestudio. 
## Dreambooth3D Approach
The Dreambooth3D approach is split into 3 stages - 
### Stage 1
A DreamBooth model which comprises of a text to image stable diffusion model is trained using 5-6 casually captured images of a given subject. In order to generate subject specific images, the text prompt uses a unique identifier along with the class it belongs to. 
To train the dreambooth model you can run the following command 
```sh
# download the subject images - for example to download the dog images
python3 download_dog.py
# to run dreambooth training script
cd threestudio/scripts/
python3 train_dreambooth_script.py --instance_dir <path to images> --instance_prompt <prompt_text with identifier> --output_dir <output_dir> --class_dir <where class images will be stored> --class_prompt <prompt for class>
python3 train_dreambooth_script.py --instance_dir ./dog --instance_prompt "Photo of sks dog" --output_dir ./saved_model_dog --class_dir ./class_folder --class_prompt "Photo of a dog"
```
The partially trained DreamBooth model is then combined with the DreamFusion which performs NeRF optimisation using Score Distillation Sampling (SDS) loss in order to generate 3D assets. We use an intermediate checkpoint of the dreambooth model (We have chosen checkpoint-500 as the optimal checkpoint based on our experiments). Ensure that the config file has the parameters set correctly (set first_stage to true, set the pretrained_model_name_or_path to the path to the model and checkpoint). To train the dreamfusion model run the following command
```
python launch.py --config <config_file> --train --gpu 0 system.prompt_processor.prompt=<prompt>
# example
python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a photo of sks dog"
```
Stage 1 is trained for 10000 iterations.
### Stage 2
In this stage  multi-view renders along random viewpoints are obtained using NeRF in DreamFusion. Here, we use Img2Img translation to generate the pseudo multi-view images which cover a wide range of views and maintain high subject-specific likeness. Here, we use the fully-trained Dreambooth model (checkpoint-800). The strength denotes the amount of noise to be added to the input images in image in image to image translation (0 - no noise, 1 - complete noise). The optimal value based on our experiments is in the range 0.3-0.5. To generate the pseudo-images run the following command
```sh
python threestudio/scripts/dreambooth3d_img2img.py --pretrained_model_name_or_path <path to fully trained checkpoint> --prompt <prompt_text> --input_folder <folder_containing_images_sampled_in_stage1> --strength <strength_value>
```

### Stage 3
The fully-trained DreamBooth model is then used to optimize the DreamFusion model further using SDS (Score distillation sampling).  In order to account for color shift a reconstruction loss is used in this stage on the pseudo images generated in stage 2. To run stage 3 execute the following command with the updated config file (set first_stage to False, provide custom_data_path (path to images generated in stage 2)).

```sh
python launch.py --config <config_file> --train --gpu 0 system.prompt_processor.prompt=<prompt>
# example
python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a photo of sks dog"
```
Stage 3 is trained for 5000 iterations.

### Run all 3 stages together
To run all the three stages for the dreambooth3d-sd.yaml (dog) (after obtaining the dreambooth checkpoints) use the following script
```sh
python run_3_stages.py --partial_model_name_or_path <path_to_partial_ckpt> --pretrained_model_name_or_path <path_to_model_dir> --fully_trained_model_name_or_path <path_to_fully_trained_ckpt> --strength <strength_value> --prompt <prompt_text>
# example
python run_3_stages.py --partial_model_name_or_path dreambooth_weights/checkpoint-500/ --pretrained_model_name_or_path dreambooth_weights/ --fully_trained_model_name_or_path dreambooth_weights/checkpoint-800/ --strength 0.3 --prompt "a photo of sks dog"
```
### Visualizations

You can find visualizations of the current status in the trial directory which defaults to `[exp_root_dir]/[name]/[tag]@[timestamp]`, where `exp_root_dir` (`outputs/` by default), `name` and `tag` can be set in the configuration file. A 360-degree video will be generated after the training is completed. In training, press `ctrl+c` one time will stop training and head directly to the test stage which generates the video. Press `ctrl+c` the second time to fully quit the program.

### Code Structure

Brief introduction of the codestructure of the threestudio framework (As provided by the original repo).

- All methods are implemented as a subclass of `BaseSystem` (in `systems/base.py`). There typically are six modules inside a system: geometry, material, background, renderer, guidance, and prompt_processor. All modules are subclass of `BaseModule` (in `utils/base.py`) except for guidance, and prompt_processor, which are subclass of `BaseObject` to prevent them from being treated as model parameters and better control their behavior in multi-GPU settings.
- All systems, modules, and data modules have their configurations in their own dataclasses.
- Base configurations for the whole project can be found in `utils/config.py`. In the `ExperimentConfig` dataclass, `data`, `system`, and module configurations under `system` are parsed to configurations of each class mentioned above. These configurations are strictly typed, which means you can only use defined properties in the dataclass and stick to the defined type of each property. This configuration paradigm (1) natually supports default values for properties; (2) effectively prevents wrong assignments of these properties (say typos in the yaml file) or inappropriate usage at runtime.

### Our Contributions to the Code
- Added the threestudio/scripts/train_dreambooth.py and threestudio/scripts/train_dreambooth_script.py for training the dreambooth model for Stage 1.
- Added the threestudio/systems/dreambooth3d.py script based off of the dreamfusion.py including our own dataset class. Also added the reconstruction loss for Stage 3. 
- Added own dataset class in threestudio/data/multi_view_dreambooth3d.py for storing and loading random-viewpoints for Stage 2 and 3.
- Added the  threestudio/scripts/dreambooth3d_img2img.py for Img2Img translation for Stage 2. 
- Other changes to combine the code were made in the threestudio/data and threestudio/utils backend files and threestudio/models/guidance/stable_diffusion_guidance.py. 
- We have also added config files for parameter-tuning for each of the prompts in the config folder.

### Dreambooth Checkpoints 
The already trained dreambooth checkpoints can be downloaded using the following links for each of the prompts
- [Dog Checkpoint](https://drive.google.com/drive/folders/1uCMWirJosk7w_sF3ehsYCBhkxYH-rlsp?usp=drive_link) - [Images](https://drive.google.com/drive/folders/1YH7UQno8PGXcbw38Sy-94xUTAQ983pLY?usp=drive_link)
```sh
gdown --folder https://drive.google.com/drive/folders/1uCMWirJosk7w_sF3ehsYCBhkxYH-rlsp?usp=drive_link
```
- [Cat Checkpoint](https://drive.google.com/drive/folders/19C5flYMdfyGGcUxdqPtYeT9pffZ8ElwY?usp=drive_link) - [Images](https://drive.google.com/drive/folders/1VO3omjGxyngDhHWtBFAh0SmcQbdk-p9h?usp=drive_link)
```sh
gdown --folder https://drive.google.com/drive/folders/19C5flYMdfyGGcUxdqPtYeT9pffZ8ElwY?usp=drive_link
```
- [Clock Checkpoint](https://drive.google.com/drive/folders/1ETWUXvBP8EuIHMKP5B4MADRi44YCBbM9?usp=drive_link) - [Images](https://drive.google.com/drive/folders/1BrL7MpKAeHIEBK1_TkQcgjop7W1EBHnF?usp=drive_link)
```sh
gdown --folder https://drive.google.com/drive/folders/1ETWUXvBP8EuIHMKP5B4MADRi44YCBbM9?usp=drive_link
```
- [Can Checkpoint](https://drive.google.com/drive/folders/1OBwIW88LcGjTKh_x1OAhT9qZQnSMxKxo?usp=sharing) - [Images](https://drive.google.com/drive/folders/1eLLlnxJF2t7bUn1UcNsAv96y5RnyTppV?usp=drive_link)
```sh
gdown --folder https://drive.google.com/drive/folders/1OBwIW88LcGjTKh_x1OAhT9qZQnSMxKxo?usp=sharing
```
## Team Members
- Aishwarya Manjunath
- Jaidev Shriram
- Niraj Mahajan
- Shreya Sumbetla

## Credits

threestudio is built on the following amazing open-source projects:

- **[Lightning](https://github.com/Lightning-AI/lightning)** Framework for creating highly organized PyTorch code.
- **[OmegaConf](https://github.com/omry/omegaconf)** Flexible Python configuration system.
- **[NerfAcc](https://github.com/KAIR-BAIR/nerfacc)** Plug-and-play NeRF acceleration.

The following repositories greatly inspire threestudio:

- **[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion)**
- **[Latent-NeRF](https://github.com/eladrich/latent-nerf)**
- **[Score Jacobian Chaining](https://github.com/pals-ttic/sjc)**
- **[Fantasia3D.unofficial](https://github.com/ashawkey/fantasia3d.unofficial)**

Thanks to the maintainers of these projects for their contribution to the community!

## Citing threestudio

If you find threestudio helpful, please consider citing:

```
@Misc{threestudio2023,
  author =       {Yuan-Chen Guo and Ying-Tian Liu and Chen Wang and Zi-Xin Zou and Guan Luo and Chia-Hao Chen and Yan-Pei Cao and Song-Hai Zhang},
  title =        {threestudio: A unified framework for 3D content generation},
  howpublished = {\url{https://github.com/threestudio-project/threestudio}},
  year =         {2023}
}
```
