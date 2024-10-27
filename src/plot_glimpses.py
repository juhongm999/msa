import os
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from torchvision.ops import box_convert


def denormalize(v, m, r):
    return (((v - m[0]) / (m[1] - m[0])) * (
        r[1] - r[0]) + r[0]).float()


def denormalize2(H, coords):
    """Convert coordinates in the range [-1, 1] to
    coordinates in the range [0, H] where `H` is
    the size of the image.
    """
    return (0.5 * ((coords + 1.0) * H)).long()


def load_data(config, test, epoch):
    model_name = config['model_name']
    plot_dir = config['plot_dir']
    g = config['g']

    output_npz_file = 'test.npz' if test else f'val_{epoch}.npz'
    saved_data = dict(np.load(os.path.join(
        plot_dir, model_name, output_npz_file)))

    glimpses = saved_data['imgs']
    locations = saved_data['locs']
    if config['use_memory']:
        attention = saved_data['attn']
    else:
        attention = None
    output = saved_data['out']

    if glimpses.shape[1] == 1:  # remove single channel dim, MNIST
        glimpses = np.squeeze(glimpses, axis=1)

    num_anims = locations.shape[1]
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[2]

    # m = (-1, 1)
    # ls = denormalize(torch.Tensor(locations), m, (0, img_shape))
    # wh = torch.tensor([g]).repeat(num_cols, 2)
    ls = denormalize2(img_shape, torch.Tensor(
        locations)).transpose(0, 1) - g//2
    wh = torch.tensor([g]).repeat(
        num_cols, 2).unsqueeze(0).repeat(num_anims, 1, 1)
    boxes_per_anim = torch.cat([ls, wh], dim=2).numpy()

    # boxes_per_anim = []
    # for i in range(num_anims):
    #     box = torch.hstack([ls[:, i], wh]).numpy()
    #     # box = box_convert(  # note: out_fmt differs from retina code
    #     #     boxes, in_fmt='cxcywh', out_fmt='xywh').numpy()
    #     boxes_per_anim.append(box)
    # boxes_per_anim = np.array(boxes_per_anim)
    # print(ls[0], boxes_per_anim[0])

    return glimpses, boxes_per_anim, attention, output


def animate(config, test, epoch):
    dataset = 'test' if test else f'val_epoch_{epoch}'
    print(f'Making *animation* plot for {dataset}...')

    model_name = config['model_name']
    plot_dir = config['plot_dir']

    glimpses, boxes_per_anim, attention, output = load_data(
        config, test, epoch)

    fig, axs = plt.subplots(nrows=1, ncols=glimpses.shape[0])

    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap='gray_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def updateData(i):
        box = boxes_per_anim[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            rect = patches.Rectangle(
                box[j, :2], box[j, 2], box[j, 3],
                linewidth=1, edgecolor='red', fill=False
            )
            ax.add_patch(rect)

    anim = animation.FuncAnimation(
        fig, updateData, frames=boxes_per_anim.shape[0], interval=500, repeat=True
    )
    anim.save(os.path.join(plot_dir, model_name, dataset + '.gif'))


def grid(config, test, epoch, pdf, class_index):
    dataset = 'test' if test else f'val_{epoch}'
    print(f'Making *grid* plot for {dataset}...')

    font_size = 9

    model_name = config['model_name']
    plot_dir = config['plot_dir']

    glimpses, boxes_per_anim, attention, output = load_data(
        config, test, epoch)

    # sort by class label
    if class_index >= 0:
        inds = np.where(output[:, 0] == class_index)[0]
    else:
        inds = np.argsort(output[:, 0])
    glimpses = glimpses[inds]
    boxes_per_anim = boxes_per_anim[:, inds]
    output = output[inds]

    nrows, ncols = glimpses.shape[0], boxes_per_anim.shape[0]

    additional = 0
    if config['use_memory']:
        attention = attention[inds]
        additional = attention.shape[1]
        if additional > 1:
            additional += 1
        vmin, vmax = attention.min(), attention.max()

    fig, axs = plt.subplots(nrows, ncols + additional,
                            figsize=(ncols + additional, max(nrows, 2)))
    if class_index >= 0:
        axs = np.array([axs])
    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i, j]
            ax.imshow(
                glimpses[i],
                cmap='gray_r' if (config['data_name'] in ['mnist', 'cluttered']) else 'gray')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            box = boxes_per_anim[j, i]
            rect = patches.Rectangle(
                box[:2], box[2], box[3],
                linewidth=1, edgecolor='red', fill=False
            )
            ax.add_patch(rect)
            if i == 0:
                ax.set_title(f'$t-{ncols-j-1}$', fontsize=8)

        if config['use_memory']:
            for k in range(attention.shape[1]):
                ax = axs[i, k+ncols]
                ax.imshow(attention[i, k], cmap='inferno',
                          vmin=0, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title(f'$\mathbf{{w}}^{k}$', fontsize=font_size)
            if additional > 1:
                ax = axs[i, -1]
                ax.imshow(np.mean(attention[i], axis=0),
                          cmap='inferno', vmin=0, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title(r'$\bar{\mathbf{{w}}}$', fontsize=font_size)

        axs[i, 0].set_ylabel(
            f'T={output[i, 0]}, Y={output[i, 1]}', fontsize=font_size)

    fig.tight_layout()

    fig.savefig(os.path.join(plot_dir, model_name, dataset + (str(class_index) if class_index >= 0 else '')+('.pdf' if pdf else '.png')),
                bbox_inches='tight', dpi=300)

    pass


if __name__ == "__main__":
    """
    Usage:
    python plot_glimpses.py -c ../configs/__config__.yaml [--test | --epoch n]
    """

    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument('-c', '--config', metavar='path', type=str,
                        required=True, help='the path to config file')

    # default false, grid view...
    parser.add_argument('--animate', action='store_true', help='animate')

    # default false, training mode...
    parser.add_argument('--test', action='store_true', help='test mode')

    parser.add_argument('--pdf', action='store_true', help='save as pdf')

    parser.add_argument('-i', '--class-index', type=int, required=False, default=-1,
                        help="desired class index")

    # only use for val data
    parser.add_argument("--epoch", type=int, required=False, default=1,
                        help="epoch of desired plot")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.animate:
        animate(config, test=args.test, epoch=args.epoch)
    else:
        grid(config, test=args.test, epoch=args.epoch,
             pdf=args.pdf, class_index=args.class_index)
