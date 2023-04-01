import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from utils import dist_util, logger
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from PIL import Image
from attrdict import AttrDict
import yaml


def img_pre_pros(img_path):
    pil_image = Image.open(img_path)
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/test_cfg.yaml',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = AttrDict(create_cfg(cfg))
    model_path = cfg.model_path
    sty_img_path = cfg.sty_img_path
    total_txt_file = cfg.total_txt_file
    gen_txt_file = cfg.gen_txt_file
    img_save_path = cfg.img_save_path
    classifier_free = cfg.classifier_free
    cont_gudiance_scale = cfg.cont_scale
    sk_gudiance_scale = cfg.sk_scale
    cfg.__delattr__('model_path')
    cfg.__delattr__('sty_img_path')
    cfg.__delattr__('total_txt_file')
    cfg.__delattr__('gen_txt_file')
    cfg.__delattr__('img_save_path')
    cfg.__delattr__('classifier_free')
    cfg.__delattr__('cont_scale')
    cfg.__delattr__('sk_scale')

    dist_util.setup_dist()

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if cfg.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    noise = None

    # gen txt
    char2idx = {}
    with open(total_txt_file, 'r') as f:
        chars = f.readlines()
        for idx, char in enumerate(chars[0]):
            char2idx[char] = idx
        f.close()
    char_idx = []
    with open(gen_txt_file, 'r') as f1:
        genchars = f1.readlines()
        for char in genchars[0]:
            char_idx.append(char2idx[char])
        f1.close()

    all_images = []
    all_labels = []

    ch_idx = 0
    while len(all_images) * cfg.batch_size < cfg.num_samples:
        model_kwargs = {}
        classes = th.tensor([i for i in char_idx[ch_idx:ch_idx + cfg.batch_size]], device=dist_util.dev())
        ch_idx += cfg.batch_size

        model_kwargs["y"] = classes
        img = th.tensor(img_pre_pros(sty_img_path), requires_grad=False).cuda().repeat(cfg.batch_size, 1, 1, 1)
        sty_feat = model.sty_encoder(img)
        model_kwargs["sty"] = sty_feat
        if cfg.stroke_path is not None:
            chars_stroke = th.empty([0, 32], dtype=th.float32)
            with open(cfg.stroke_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    strokes = line.split(" ")[1:-1]
                    char_stroke = []
                    for stroke in strokes:
                        char_stroke.append(int(stroke))
                    while len(char_stroke) < 32:  # for korean
                        char_stroke.append(0)
                    assert len(char_stroke) == 32
                    chars_stroke = th.cat((chars_stroke, th.tensor(char_stroke).reshape([1, 32])), dim=0)
            f.close()
            model_kwargs["stroke"] = chars_stroke[classes].to(dist_util.dev())
        if classifier_free:
            if cfg.stroke_path is not None:
                model_kwargs["mask_y"] = th.cat([th.zeros([cfg.batch_size], dtype=th.bool), th.ones([cfg.batch_size * 2], dtype=th.bool)]).to(dist_util.dev())
                model_kwargs["y"] = model_kwargs["y"].repeat(3)
                model_kwargs["mask_stroke"] = th.cat(
                    [th.ones([cfg.batch_size], dtype=th.bool),th.zeros([cfg.batch_size], dtype=th.bool), th.ones([cfg.batch_size], dtype=th.bool)]).to(
                    dist_util.dev())
                model_kwargs["stroke"] = model_kwargs["stroke"].repeat(3, 1)
                model_kwargs["sty"] = model_kwargs["sty"].repeat(3, 1)
            else:
                model_kwargs["mask_y"] = th.cat([th.zeros([cfg.batch_size], dtype=th.bool), th.ones([cfg.batch_size], dtype=th.bool)]).to(dist_util.dev())
                model_kwargs["y"] = model_kwargs["y"].repeat(2)
                model_kwargs["sty"] = model_kwargs["sty"].repeat(2, 1)
        else:
            model_kwargs["mask_y"] = th.zeros([cfg.batch_size], dtype=th.bool).to(dist_util.dev())
            if cfg.stroke_path is not None:
                model_kwargs["mask_stroke"] = th.zeros([cfg.batch_size], dtype=th.bool).to(dist_util.dev())

        def model_fn(x_t, ts, **model_kwargs):
            if classifier_free:
                repeat_time = model_kwargs["y"].shape[0] // x_t.shape[0]
                x_t = x_t.repeat(repeat_time, 1, 1, 1)
                ts = ts.repeat(repeat_time)

                if cfg.stroke_path is not None:
                    model_output = model(x_t, ts, **model_kwargs)
                    model_output_y, model_output_stroke, model_output_uncond = model_output.chunk(3)
                    model_output = model_output_uncond + \
                                   cont_gudiance_scale * (model_output_y - model_output_uncond) + \
                                   sk_gudiance_scale * (model_output_stroke - model_output_uncond)

                else:

                    model_output = model(x_t, ts, **model_kwargs)
                    model_output_cond, model_output_uncond = model_output.chunk(2)
                    model_output = model_output_uncond + cont_gudiance_scale * (model_output_cond - model_output_uncond)

            else:
                model_output = model(x_t, ts, **model_kwargs)
            return model_output

        sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (cfg.batch_size, 3, cfg.image_size, cfg.image_size),
            clip_denoised=cfg.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = [
            th.zeros_like(classes) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * cfg.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: cfg.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: cfg.num_samples]
    if dist.get_rank() == 0:
        for idx, (img_sample, img_cls) in enumerate(zip(arr, label_arr)):
            img = Image.fromarray(img_sample).convert("RGB")
            img_name = "%05d.png" % (idx)
            img.save(os.path.join(img_save_path, img_name))

    dist.barrier()
    logger.log("sampling complete")


def create_cfg(cfg):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cont_scale=1.0,
        sk_scale=1.0,
        sty_img_path="",
        stroke_path=None,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()