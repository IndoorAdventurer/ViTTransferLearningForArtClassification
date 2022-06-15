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
getmodel_fun = lp.get_swin_b_problem if args.ots else lp.get_swin_b_drop_problem
model, dl = getmodel_fun(off_the_shelf=False, pretrained=False, dl=datloader)
model.load_state_dict(torch.load(args.state_dict, map_location=torch.device('cpu')))
model = model.eval()

#---GETTING-ATTENTION-WEIGHTS:-----------------------------------------------------
att_weighs = []
def add_to_att_weights(module, inp, outp):
    global att_weighs
    # Taking the mean of all attention heads, but could also be max or min
    att_weighs += [outp.detach().numpy()]

for name, module in model.named_modules():
    if "attn_drop" in name: # A timm thing. This only works for timm models ;-)
        module.register_forward_hook(add_to_att_weights)

# Getting the image and passing it through the model:
image = io.read_image(args.image, mode = io.ImageReadMode.RGB).float() / 255
image = defs.buildTransform(imnet_norm=True)(image).unsqueeze(0)
prediction = model(image)

# Saving prediction to file
prediction = dl.materials[torch.argmax(prediction)]
with open(args.target + "_prediction.txt", "w") as f:
    f.write(prediction)


#---CHANGING-ATTENTION-WEIGHTS-TO-VIT-STYLE:---------------------------------------
def meanAttentionPerLayer(att_mat, shift=0):  
    # Keeping the dimension in case I want to do something with it later:
    m = np.mean(att_mat, axis=1, keepdims=True)

    batch, head, frm, to = m.shape
    
    sqrt_b = int(np.sqrt(batch))

    m = m.reshape((sqrt_b, sqrt_b, head, frm, to))

    # Turning it into a ViT-like attention structure, with [heads, from, to]
    # instead of [window, head, win_from, win_to]:
    all_rows = []
    zero_layer = np.zeros((head, 49, 7 * sqrt_b, 7 * sqrt_b))
    for row_idx, row in enumerate(m):
        for col_idx, col in enumerate(row):
            tmp = zero_layer.copy()
            tmp[:, :, row_idx * 7: row_idx * 7 + 7, col_idx * 7: col_idx * 7 + 7] = col.reshape((head, 49, 7,7))
            all_rows += [tmp]
    m = np.concatenate(all_rows, axis=1)
    del all_rows

    # Inverting the window shifts:
    if shift != 0:
        m = np.roll(m, shift=shift, axis=(2,3))

    # Currently: heads, frm, sqrt_to, sqrt_to
    # Making it first, heads, sqrt_frm, sqrt_frm, sqrt_to, sqrt_to:
    s = m.shape
    repeats = int((3136 / s[1]) ** .5) # Already for next step
    sqrt_frm = int(s[1] ** .5)
    m = m.reshape(s[0], sqrt_frm, sqrt_frm, s[2], s[3])

    # Expanding it such that it is the same size as the final one
    # if repeats > 1:
    #     m = m.repeat(repeats=repeats, axis=1)     REMOVED THIS BECAUSE
    #     m = m.repeat(repeats=repeats, axis=2)     I AM NO LONGER DOING
    #     m = m.repeat(repeats=repeats, axis=3)     ATTENTION ROLLOUT
    #     m = m.repeat(repeats=repeats, axis=4)     FOR SWIN! GRADCAM INSTEAD

    # Change the shape to how it works for ViTs as well
    s = m.shape
    m = m.reshape(s[0], s[1] * s[2], s[3] * s[4])
    m /= m.sum(axis=2, keepdims=True)

    return m.mean(axis=0)

#---PROCESSING-AND-SAVING-LAYERS:--------------------------------------------------
print("Processing layers", flush=True)
with open(args.target + "_layers.npy", "wb") as f:
    for idx, layer in enumerate(att_weighs):
        A = meanAttentionPerLayer(layer, shift= 3 * int(idx % 2))
        np.save(f, A)
        print(".", end="", flush=True)

print("\nThe script is done!!!")