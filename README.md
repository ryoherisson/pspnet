# PSPNet: Pyramid Scene Parsing Network in pytorch
This is a pytorch implementation of Pyramid Scene Parsing Network in pytorch.  

(Reference)  
https://github.com/YutaroOgawa/pytorch_advanced/tree/master/3_semantic_segmentation

## Requirements
```bash
$ pip install -r requirements.txt
```

## Usage
### Configs
Create a configuration file based on configs/default.yaml.
```bash
# ----------
# dataset
# ----------
data_root: ./dataset/
n_classes: 21
classes:    ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
img_size: 475
n_channels: 3
color_mean: [0.485, 0.456, 0.406]
color_std: [0.229, 0.224, 0.225]
train_txt: train.txt
test_txt: val.txt
img_extension: .jpg
anno_extension: .png

# ----------------
# train parameters
# ----------------
lr: 0.0001
decay: 0.0001
n_gpus: 2 # currently works with only one gpu
batch_size: 64
n_epochs: 50
input_size_8: 60
aux_weight: 0.4

# save_ckpt_interval should not be 0.
save_ckpt_interval: 50

# output dir (logs, results)
log_dir: ./logs/

# checkpoint path or blank
resume: 
# e.g) resume: ./logs/2020-07-26T00:19:34.918002/ckpt/best_iou_ckpt.pth

# ----------------
# Visualize Results
# ----------------
# visualize label color_map
label_color_map: [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
                ]
```

### Prepare Dataset
If you want to use your own dataset, you need to prepare a directory with the following structure:
```bash
datasets/
├── annotations
│   ├── hoge.jpg
│   ├── fuga.jpg
│   ├── foo.jpg
│   └── bar.jpg
├── images
│   ├── hoge.jpg
│   ├── fuga.jpg
│   ├── foo.jpg
│   └── bar.jpg
├── train.txt
└── test.txt
```

The content of the txt file should have the following structure.
```bash
hoge
fuga
foo
bar
```

An example of a custom dataset can be found in the dataset folder.

### Train
```bash
$ python main.py --config ./configs/default.yaml
```

### Inference
```bash
$ python main.py --config ./configs/default.yaml --inference
```

### Tensorboard
```bash
tensorboard --logdir={log_dir} --port={your port}
```
![tensorboard](docs/images/tensorboard.png)

## Output
You will see the following output in the log directory specified in the Config file.
```bash
# Train
logs/
└── 2020-07-26T14:21:39.251571
    ├── checkpoint
    │   ├── best_acc_ckpt.pth
    │   ├── epoch0000_ckpt.pth
    │   └── epoch0001_ckpt.pth
    ├── metrics
    │   └── train_metrics.csv 
    │   └── test_metrics.csv 
    ├── tensorboard
    │   └── events.out.tfevents.1595773266.c47f841682de
    └── logfile.log

# Inference
inference_logs/
└── 2020-07-26T14:21:06.197407
    ├── images
    │   └── hoge.jpg 
    ├── metrics
    │   └── test_metrics.csv 
    ├── tensorboard
    │   └── events.out.tfevents.1595773266.c47f841682de
    └── logfile.log
```

The contents of train_metrics.csv and test_metrics.csv look like as follows:
```bash
epoch, train loss, train mean iou
0,3.8158764839172363,0.2572
1,3.4702939987182617,0.1169
```
You can monitor loss, iou per class and mean iou in the logfile during training.
