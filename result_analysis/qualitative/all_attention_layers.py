import pandas as pd
import numpy as np
from torchvision import io
import cv2

import os
import sys
sys.path.insert(1, "/home/vincent/Documenten/BachelorsProject/GitHub_Repo/")
from rijks_torch.learning_problems.defaults import buildTransform

models = [
    (
        "get_swin_attention_layers.py",
        "/home/vincent/Documenten/BachelorsProject/best_models/ft_type1_swin_b_drop-best.pth",
        "/home/vincent/Documenten/BachelorsProject/Rijksdata/type/1/fullsize",
        False # Not off the shelf
    ),
    (
        "get_swin_attention_layers.py",
        "/home/vincent/Documenten/BachelorsProject/best_models/ots_type2_swin_b-best.pth",
        "/home/vincent/Documenten/BachelorsProject/Rijksdata/type/2/fullsize",
        True # Is off the shelf
    ),
    (
        "get_deit_attention_layers.py",
        "/home/vincent/Documenten/BachelorsProject/best_models/ft_type3_deit_b_16_drop-best.pth",
        "/home/vincent/Documenten/BachelorsProject/Rijksdata/type/3/fullsize",
        False # Not off the shelf
    ),
    (
        "get_deit_attention_layers.py",
        "/home/vincent/Documenten/BachelorsProject/best_models/ots_type3_deit_b_16-best.pth",
        "/home/vincent/Documenten/BachelorsProject/Rijksdata/type/3/fullsize",
        True # Is off the shelf
    )
]

image_path = "/home/vincent/Documenten/BachelorsProject/Rijksdata/jpg/"

images_csv = sys.argv[1]
target_dir = sys.argv[2]

df = pd.read_csv(images_csv)

for idx, img in enumerate(df["jpg"]):
    print(f"---Doing image {idx}" + 40 * "-")

    # Saving both a colored and grayscale version of the input image
    image = io.read_image(image_path + img, mode = io.ImageReadMode.RGB).float() / 255
    image = buildTransform(imnet_norm=False)(image).unsqueeze(0)[0]
    image = image.detach().numpy().transpose(1,2,0)
    np.save(os.path.join(target_dir, f"img{idx:03}"), image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    np.save(os.path.join(target_dir, f"img{idx:03}_gray"), image)

    # Saving all the layers for all models:
    for model in models:
        target = "swin" if "swin" in model[0] else "deit"
        print(f"------Doing model {target}" + 20 * "-")
        target += "_ots" if model[3] else "_ft"
        target += f"_img{idx:03}"
        target = os.path.join(target_dir, target)
        
        cmd = f"python3 {model[0]} --state_dict {model[1]} --dataset {model[2]} --image {image_path + img} --target {target}"
        if model[3]:
            cmd += " --ots"
        os.system(cmd)