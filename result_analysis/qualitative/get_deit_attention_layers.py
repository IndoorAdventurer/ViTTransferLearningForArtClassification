import sys
sys.path.insert(1, "/home/vincent/Documenten/BachelorsProject/GitHub_Repo/")
import rijks_torch.learning_problems as lp
import rijks_torch.learning_problems.defaults as defs
from rijks_torch.data_loading import RijksDataloaders

import numpy as np
import torch
from torchvision import io
import argparse


arg_parser = argparse.ArgumentParser(description="Export all Swin attention layers for given model file (.pth) and image")
arg_parser.add_argument("--state_dict", type=str, required=True)
arg_parser.add_argument("--dataset", type=str, required=True)
arg_parser.add_argument("--image", type=str, required=True)
arg_parser.add_argument("--target", type=str, required=True)
arg_parser.add_argument("--ots", action="store_true")
args = arg_parser.parse_args()

# Getting dataset and model:
datloader = RijksDataloaders(
    ds_name=args.dataset,
    hist_path=args.dataset + "-hist.csv",
    img_dir="/home/vincent/Documenten/BachelorsProject/Rijksdata/jpg/",
    transforms={"all": defs.buildTransform(imnet_norm=True)},
    batch_size=1
)
getmodel_fun = lp.get_deit_b_16_problem if args.ots else lp.get_deit_b_16_drop_problem
model, dl = getmodel_fun(off_the_shelf=False, pretrained=False, dl=datloader)
model.load_state_dict(torch.load(args.state_dict, map_location=torch.device('cpu')))
model = model.eval()

#---GETTING-ATTENTION-WEIGHTS-AND-SAVING-TO-FILE:----------------------------------
f = open(args.target + "_layers.npy", "wb")
def add_to_att_weights(module, inp, outp):
    print(".", end="", flush=True)
    np.save(f, outp.detach().numpy()[0].mean(axis=0))

for name, module in model.named_modules():
    if "attn_drop" in name: # A timm thing. This only works for timm models ;-)
        module.register_forward_hook(add_to_att_weights)

# Getting the image and passing it through the model:
image = io.read_image(args.image, mode = io.ImageReadMode.RGB).float() / 255
image = defs.buildTransform(imnet_norm=True)(image).unsqueeze(0)
prediction = model(image)
f.close()

# Saving prediction to file
prediction = dl.materials[torch.argmax(prediction)]
with open(args.target + "_prediction.txt", "w") as f:
    f.write(prediction)

print("\nThe script is done!!!")