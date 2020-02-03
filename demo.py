import os

# %%
import matplotlib.pyplot as plt
import matplotlib.image
import plac

import loaddata_demo as loaddata
from custom_transforms import *
from nn_model import Net

plt.set_cmap("gray")


@plac.annotations(
    image_path=plac.Annotation('The path to an RGB image or a directory containing RGB images.', type=str,
                               kind='option', abbrev='i'),
    model_path=plac.Annotation('The path to the pre-trained model weights.', type=str, kind='option', abbrev='m'),
    output_path=plac.Annotation('The path to save the model output to.', type=str, kind='option', abbrev='o'),
)
def main(image_path, model_path='all-scales-trained.ckpt', output_path=None):
    print("Loading model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Creating depth maps...")
    rgb_path = os.path.abspath(image_path)

    if os.path.isdir(rgb_path):
        for file in os.listdir(rgb_path):
            test(model, os.path.join(rgb_path, file), output_path)
    else:
        test(model, rgb_path, output_path)

    print("Done.")


def test(model, rgb_path, output_path):
    nyu2_loader = loaddata.readNyu2(rgb_path)

    path, file = os.path.split(rgb_path)
    file = f"{file.split('.')[0]}.png"
    depth_path = os.path.join(output_path, file) if output_path else os.path.join(path, f"out_{file}")

    print(f"{rgb_path} -> {depth_path}")

    for i, image in enumerate(nyu2_loader):
        image = image.cuda()
        out = model(image)

        matplotlib.image.imsave(depth_path, out.view(out.size(2), out.size(3)).data.cpu().numpy())


if __name__ == '__main__':
    plac.call(main)
