from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
from fontTools.ttLib import TTFont


parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='../ttf_folder',help='ttf directory')
parser.add_argument('--chara', type=str, default='../chara.txt',help='characters')
parser.add_argument('--save_path', type=str, default='../save_folder',help='images directory')
parser.add_argument('--img_size', type=int, help='The size of generated images')
parser.add_argument('--chara_size', type=int, help='The size of generated characters')
args = parser.parse_args()

file_object = open(args.chara,encoding='utf-8')   
try:
	characters = file_object.read()
finally:
    file_object.close()


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(data_root)

all_image_paths = list(data_root.glob('*.*'))  # *.ttf TTF
all_image_paths = [str(path) for path in all_image_paths]
total_num = len(all_image_paths)
print(total_num)

seq = list()

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

def get_char_list_from_ttf(font_file):
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()

    unicode_list = []
    for key, uni in m_dict.items():
        unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

for idx, (label, item) in enumerate(zip(range(len(all_image_paths)),all_image_paths)):
    print("{} / {} ".format(idx, total_num), item)
    src_font = ImageFont.truetype(item, size=args.chara_size)
    font_name = item.split('/')[-1].split('.')[0]
    chars = get_char_list_from_ttf(item)  #
    img_cnt = 0
    for (chara, cnt) in zip(characters, range(len(characters))):
        img = draw_example(chara, src_font, args.img_size, (args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2)
        path_full = os.path.join(args.save_path, 'id_%d'%(label))
        if not os.path.exists(path_full):
            os.mkdir(path_full)
        if np.sum(np.array(img) / 255.) < 18000:
            img_cnt += 1
            img.save(os.path.join(path_full, "%05d.png" % (cnt)))


