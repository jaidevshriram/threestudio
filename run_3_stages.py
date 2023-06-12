# script to run all 3 stages 
import argparse
import os
import yaml

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--partial_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to partially trained dreambooth model",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to dreambooth model",
    )

    parser.add_argument(
        "--fully_trained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to fully trained dreambooth model",
    )
    # add argument for strength
    parser.add_argument(
        "--strength",
        type=float,
        default=0.25,
        required=False,
        help="Strength for image generation.",
    )

    # add argument for prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Prompt for image generation.",
    )


    if input_args is not None:
            args = parser.parse_args(input_args)
    else:
            args = parser.parse_args()

    return args

# take in arguments from command line
args = parse_args()

# update the configs/dreambooth3d-sd.yaml to change the first_stage argument to True
#"""
output_folder_name = "dreambooth3d-sd"
datalist = None
with open("configs/dreambooth3d-sd.yaml", 'r') as f:
        data = yaml.safe_load_all(f)
        #print("Original config file:")
        #print(list(data))
        datalist = list(data)
        #print(datalist)
        #print(type(datalist))
        for doc in datalist:
            #print("IN LOOOOOOOOOOOOOOOOP")
            #print(doc)
            for key, value in doc.items():
                #print("key: " + str(key))
                if key == 'first_stage':
                    doc[key] = True
                    #print("updated first_stage: " + str(doc[key]))
                if key == 'data':
                    doc[key]['first_stage'] = True
                    #print("updated first_stage: " + str(doc[key]))
                if key == 'system':
                    doc[key]['prompt_processor']['pretrained_model_name_or_path'] = '\"'+args.pretrained_model_name_or_path+'\"'
                    #print("updated pretrained_model_name_or_path: " + str(doc[key]))
                    doc[key]['guidance']['pretrained_model_name_or_path'] = '\"'+args.partial_model_name_or_path+'\"'
                if key == 'trainer':
                    doc[key]['max_steps'] = 5000
                    #print("updated max_steps: " + str(doc[key]))
                if key == 'name':
                    output_folder_name = doc[key]
        #print("Updated config file:")
        #print(datalist)
        #yaml.dump_all(data, f, sort_keys=False)

# dump updated data to same file
with open("configs/dreambooth3d-sd.yaml", 'w') as f:
    yaml.dump_all(datalist, f, sort_keys=False)
#"""
# remove all occurences of the character ' from the file
#"""
data = None
with open("configs/dreambooth3d-sd.yaml", 'r') as f:
    data = f.read()
    data = data.replace("'", "")
with open("configs/dreambooth3d-sd.yaml", 'w') as f:
    f.write(data)
#"""
#exit()
#"""
"""
lines = []
with open("configs/dreambooth3d-sd.yaml", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if "first_stage" in lines[i]:
            lines[i] = "  first_stage: True\n"
        # set the pretrained_model_name_or_path to the partial model
        if "pretrained_model_name_or_path" in lines[i]:
            lines[i] = "    pretrained_model_name_or_path: \"" + args.partial_model_name_or_path + "\"\n"
        # set max_steps to 10000
        if "max_steps" in lines[i]:
            lines[i] = "  max_steps: 100\n"
            break
# save the changes in the file
with open("configs/dreambooth3d-sd.yaml", "w") as f:
    f.writelines(lines)
"""
#"""
# print all args
print("partial_model_name_or_path: " + args.partial_model_name_or_path)
print("pretrained_model_name_or_path: " + args.pretrained_model_name_or_path)
print("strength: " + str(args.strength))
print("prompt: " + args.prompt)

# print("Updated config file:")
# with open("configs/dreambooth3d-sd.yaml", 'r') as f:
#         data = yaml.safe_load_all(f)
#         print(list(data))
#exit()
# run stage 1
# Run the following command in the terminal: python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=$prompt
# where $prompt is the prompt from args (e.g. "a photo of a cat")
print("Running stage 1...")
print("python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=\"" + args.prompt + "\"")
os.system("python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=\"" + args.prompt + "\"")
#print("python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=" + args.prompt)
# get input folder
# iterate through folder and get the folder name of the last folder alphabetically
# this is the folder that contains the images that were just generated
#search_folder = "outputs/dreambooth3d-sd/"
search_folder = "outputs/"+output_folder_name+"/"
last_folder_path = ""
sorted_search_folder = sorted(os.scandir(search_folder), key=lambda e: e.name)
#with sorted_search_folder as it:
for entry in sorted_search_folder:
    if entry.is_dir():
        # get path of the folder
        last_folder_path = os.path.join(search_folder, entry.name)
        #last_folder = entry.name
last_image_folder_path = ""
sorted_search_folder_img = sorted(os.scandir(last_folder_path+'/save/'), key=lambda e: e.name)
#with sorted_search_folder_img as it:
for entry in sorted_search_folder_img:
    if entry.is_dir():
        # get path of the folder
        last_image_folder_path = os.path.join(last_folder_path+'/save/', entry.name)
        #last_folder = entry.name
#print("last_image_folder_path: " + last_image_folder_path)
# stage 2
last_image_folder_path = last_image_folder_path + "/"
print("Running stage 2...")
# Run python threestudio/scripts/dreambooth3d_img2img.py --pretrained_model_name_or_path $fully_pretrained_path --prompt $prompt --input_folder $input_folder
# where $fully_pretrained_path is the path from args with strength from args
#print("python threestudio/scripts/dreambooth3d_img2img.py --pretrained_model_name_or_path " + args.pretrained_model_name_or_path + " --prompt " + args.prompt + " --input_folder " + last_image_folder_path + " --strength " + str(args.strength))
print("python threestudio/scripts/dreambooth3d_img2img.py --pretrained_model_name_or_path " + args.fully_trained_model_name_or_path + " --prompt \"" + args.prompt + "\" --input_folder " + last_image_folder_path + " --strength " + str(args.strength))
os.system("python threestudio/scripts/dreambooth3d_img2img.py --pretrained_model_name_or_path " + args.fully_trained_model_name_or_path + " --prompt \"" + args.prompt + "\" --input_folder " + last_image_folder_path + " --strength " + str(args.strength))

# stage 3
# change config file to set first_stage to False
#data = None
#path_with_q = f'"{last_image_folder_path}"'
datalist = None
with open("configs/dreambooth3d-sd.yaml", 'r') as f:
        data = yaml.safe_load_all(f)
        #print("Original config file:")
        #print(list(data))
        datalist = list(data)
        #print(datalist)
        #print(type(datalist))
        for doc in datalist:
            #print("IN LOOOOOOOOOOOOOOOOP")
            #print(doc)
            for key, value in doc.items():
                #print("key: " + str(key))
                if key == 'first_stage':
                    doc[key] = False
                if key == 'data':
                    doc[key]['first_stage'] = False
                    #print("updated first_stage: " + str(doc[key]))
                    doc[key]['custom_data_path'] = "\""+last_image_folder_path+"\""
                if key == 'system':
                    doc[key]['prompt_processor']['pretrained_model_name_or_path'] = '\"'+args.pretrained_model_name_or_path+'\"'
                    #print("updated pretrained_model_name_or_path: " + str(doc[key]))
                    doc[key]['guidance']['pretrained_model_name_or_path'] = '\"'+args.fully_trained_model_name_or_path+'\"'
                if key == 'trainer':
                    doc[key]['max_steps'] = 5000
                    #print("updated max_steps: " + str(doc[key]))
        #print("Updated config file:")
        #print(datalist)
        #yaml.dump_all(data, f, sort_keys=False)

# dump updated data to same file
with open("configs/dreambooth3d-sd.yaml", 'w') as f:
    yaml.dump_all(datalist, f, sort_keys=False)

data = None
with open("configs/dreambooth3d-sd.yaml", 'r') as f:
    data = f.read()
    data = data.replace("'", "")
with open("configs/dreambooth3d-sd.yaml", 'w') as f:
    f.write(data)
#"""
"""
lines = []
with open("configs/dreambooth3d-sd.yaml", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if "first_stage" in lines[i]:
            lines[i] = "  first_stage: False\n"
        # add last_image_folder_path to the config file for custom_data_path
        if "custom_data_path" in lines[i]:
            lines[i] = "  custom_data_path: \"" + last_image_folder_path + "\"\n"
        # set the pretrained_model_name_or_path to the fully trained model
        if "pretrained_model_name_or_path" in lines[i]:
            lines[i] = "    pretrained_model_name_or_path: \"" + args.pretrained_model_name_or_path + "\"\n"
        # set max_steps to 10000
        if "max_steps" in lines[i]:
            lines[i] = "  max_steps: 100\n"
            break

# save the changes in the file
with open("configs/dreambooth3d-sd.yaml", "w") as f:
    f.writelines(lines)
"""

print("Running stage 3...")
# run python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=$prompt
# where $prompt is the prompt from args (e.g. "a photo of a cat")
print("python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=\"" + args.prompt + "\"")
os.system("python launch.py --config configs/dreambooth3d-sd.yaml --train --gpu 0 system.prompt_processor.prompt=\"" + args.prompt + "\"")
