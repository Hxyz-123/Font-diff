## Diff-Font

------

Official code implementation based on pytorch for paper, Diff-Font: Diffusion Model for Robust One-Shot Font Generation.  [Arvix version](https://arxiv.org/pdf/2212.05895.pdf)



## Dependencies

------

```python
pytorch>=1.10.0
tqdm
opencv-python
sklearn
pillow
tensorboardX
blobfile>=1.0.5
mpi4py
attrdict
yaml
```



## Dataset

------

[方正字库](https://www.foundertype.com/index.php/FindFont/index) provides free font download for non-commercial users.

Example directory hierarchy

```python
data_dir
    |--- font1
    |--- font2
           |--- 00000.png
           |--- 00001.png
           |--- ...
    |--- ...
```



## Usage

------

### Prepare dataset

```python
python font2img.py --ttf_path ttf_folder --chara total_chn.txt --save_path save_folder --img_size 80 --chara_size 60
```

### Conditional training

- Modify the configuration file cfg/train_cfg.yaml

  Key setting for conditional training:

  ```yaml
  data_dir: 'path_to_dataset/'
  chara_nums: 6625  # num of characters
  train_steps: 420000 # conditional training steps
  sty_encoder_path: './pretrained_models/chinese_styenc.ckpt' # path to pre-trained style encoder
  model_save_dir: './trained_models' # path to save trained models
  stroke_path: './chinese_stroke.txt' # encoded strokes
  classifier_free: False # False for conditional training
  resume_checkpoint: ""
  ```

- single gpu

  ```python
  python train.py --cfg_path cfg/train_cfg.yaml
  ```

- distributed training

  ```python
  mpiexec -n $NUM_GPUS python train.py --cfg_path cfg/train_cfg.yaml
  ```

### Fine-tuning

After conditional training , we suggest an additional fine-tuning step.

- Modify the configuration file cfg/train_cfg.yaml

  Key setting for fine-tuning:

  ```yaml
  data_dir: 'path_to_dataset/'
  chara_nums: 6625  # num of characters
  model_save_dir: './trained_models' # path to save trained models
  stroke_path: './chinese_stroke.txt' # encoded strokes
  classifier_free: True  # True for fine-tuning
  total_train_steps: 800000 # total number of training steps for conditional training and fine-tuning
  resume_checkpoint: "./trained_models/model420000.pt" # path to conditional trained model, required for fine-tuning
  ```

- single gpu

  ```python
  python train.py --cfg_path cfg/train_cfg.yaml
  ```

- distributed training

  ```python
  mpiexec -n $NUM_GPUS python train.py --cfg_path cfg/train_cfg.yaml
  ```

### Test

Modify the configuration file cfg/test_cfg.yaml

Key setting for testing:

```yaml
chara_nums: 6625
num_samples: 10
stroke_path: './char_stroke.txt'
model_path: 'path_to_trained_model'
sty_img_path: 'path_to_reference_image'
total_txt_file: './total_chn.txt'
gen_txt_file: './gen_char.txt' # txt file for generation
img_save_path: './result' # path to save generated images
classifier_free: True 
cont_scale: 3.0 # content guidance sacle
sk_scale: 3.0 # stroke guidance sacle
```

, then run

```python
python sample.py --cfg_path cfg/test_cfg.yaml
```



# Acknowledgements

------

This project is based on [guided-diffusion]([openai/guided-diffusion (github.com)](https://github.com/openai/guided-diffusion)) and [DG-Font](https://github.com/ecnuycxie/DG-Font).
