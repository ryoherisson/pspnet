# PSPNet: Pyramid Scene Parsing Network in pytorch
This is a pytorch implementation of Pyramid Scene Parsing Network in pytorch.  

(Reference)  
https://github.com/YutaroOgawa/pytorch_advanced/tree/master/2_objectdetection  

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
classes: ['aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor']
img_size: 300
n_channels: 3
color_mean: [104, 117, 123]
train_txt: train.txt
test_txt: val.txt

# ----------------
# train parameters
# ----------------
lr: 0.0001
decay: 1e-4
n_gpus: 1
batch_size: 64
n_epochs: 50

# pretrained path: vgg16 model path or blank
pretrained: ./weights/vgg16_reducedfc.pth

# save_ckpt_interval should not be 0.
save_ckpt_interval: 50

# output dir (logs, results)
log_dir: ./logs/

# checkpoint path or blank
resume: ./weights/ssd300_mAP_77.43_v2.pth
# e.g) resume: ./logs/2020-07-26T00:19:34.918002/ckpt/best_acc_ckpt.pth

# visualize label color_map
label_color_map: ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
# font should be downloaded manually
font_path: ./font/calibril.ttf
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
├── train.csv
└── test.csv
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
    │   ├── hoge.jpg
    │   └── fuga.csv 
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
