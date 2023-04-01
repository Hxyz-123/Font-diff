import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from . import logger


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    stroke_path=None,
    classifier_free=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    chars_stroke = None
    class_names = [bf.basename(path).split("_")[0] for path in all_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    sty_class_names = [bf.dirname(path).split("/")[-1] for path in all_files]
    sty_sorted_classes = {x: i for i, x in enumerate(sorted(set(sty_class_names)))}
    sty_classes = [sty_sorted_classes[x] for x in sty_class_names]
    if stroke_path is not None:
        logger.log('=' * 20)
        logger.log("using strokes data")
        logger.log('=' * 20)

        chars_stroke = np.empty([0, 32], dtype=np.float32)
        with open(stroke_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                strokes = line.split(" ")[1:-1]
                char_stroke = []
                for stroke in strokes:
                    char_stroke.append(int(stroke))
                while len(char_stroke) < 32: # for korean
                    char_stroke.append(0)
                assert len(char_stroke) == 32
                chars_stroke = np.concatenate([chars_stroke, np.array(char_stroke).reshape([1, 32])], axis=0, dtype=np.float32)

    if classifier_free:
        dataset = ImageDataset_Clsfree(
            image_size,
            all_files,
            classes=classes,
            sty_classes=sty_classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            chars_stroke=chars_stroke,
        )
    else:
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            sty_classes=sty_classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            chars_stroke=chars_stroke,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        sty_classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        chars_stroke=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.style_classes = None if sty_classes is None else sty_classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.chars_stroke = chars_stroke

        if self.style_classes is not None:
            sty_classes_set = set(self.style_classes)
            self.sty_classes_idxs = {}
            for sty_class in sty_classes_set:
                self.sty_classes_idxs[sty_class] = np.where(np.array(self.style_classes) == sty_class)[0]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["mask_y"] = False
        if self.style_classes is not None:
            sty_img_idxs = self.sty_classes_idxs[self.style_classes[idx]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            with bf.BlobFile(sty_img_path, "rb") as f:
                sty_pil_image = Image.open(f)
                sty_pil_image.load()
            sty_pil_image = sty_pil_image.convert("RGB")
            sty_arr = center_crop_arr(sty_pil_image, self.resolution)
            sty_arr = sty_arr.astype(np.float32) / 127.5 - 1
            if self.chars_stroke is not None:
                out_dict["stroke"] = self.chars_stroke[self.local_classes[idx]]
                out_dict["mask_stroke"] = False
            return [np.transpose(arr, [2, 0, 1]), np.transpose(sty_arr, [2, 0, 1])], out_dict

        return np.transpose(arr, [2, 0, 1]), out_dict

class ImageDataset_Clsfree(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        sty_classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        chars_stroke=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.style_classes = None if sty_classes is None else sty_classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.chars_stroke = chars_stroke

        if self.style_classes is not None:
            sty_classes_set = set(self.style_classes)
            self.sty_classes_idxs = {}
            for sty_class in sty_classes_set:
                self.sty_classes_idxs[sty_class] = np.where(np.array(self.style_classes) == sty_class)[0]

        logger.log('=' * 20)
        logger.log("using classifier-free dataset")
        logger.log('=' * 20)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["mask_y"] = random.random() < 0.3  # 0.2
        if self.style_classes is not None:
            sty_img_idxs = self.sty_classes_idxs[self.style_classes[idx]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            with bf.BlobFile(sty_img_path, "rb") as f:
                sty_pil_image = Image.open(f)
                sty_pil_image.load()
            sty_pil_image = sty_pil_image.convert("RGB")
            sty_arr = center_crop_arr(sty_pil_image, self.resolution)
            sty_arr = sty_arr.astype(np.float32) / 127.5 - 1
            if self.chars_stroke is not None:
                out_dict["stroke"] = self.chars_stroke[self.local_classes[idx]]
                out_dict["mask_stroke"] = random.random() < 0.3
            return [np.transpose(arr, [2, 0, 1]), np.transpose(sty_arr, [2, 0, 1])], out_dict
        return np.transpose(arr, [2, 0, 1]), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
