
# DMPNet

EFFICIENT DYNAMIC MODALITY PROMPTER FOR RGBD SEMANTIC SEGMENTATION

![DMPNet](https://raw.githubusercontent.com/neilzhao996/DMPNet/refs/heads/main/figs/introduction.png)


## Trained Checkpoints
We provide trained checkpoints and prediction results 

[NYU models & results](https://drive.google.com/drive/folders/13cnfLU1SjcNGFtETgLeGl3NHWxkyrnvI?usp=drive_link), [SUNRGBD models & results](https://drive.google.com/drive/folders/1nXHLPLWHtW6FQcZfzIf75NogDf1B0vmJ?usp=drive_link)

## Usage

### Installation
```
conda create -n DMPNet python=3.9 -y
conda install pytorch==1.12.0 
```

### Data Preparation 
```
nyuv2.json and sunrgbd.json

"root": "/home/zjz/code/dataset/NYUDepthV2"   Your Path
"root": "/home/zjz/code/dataset/SUNRGBD"   Your Path

```
### Training
Put the [segformer pre-trained weight](https://github.com/NVlabs/SegFormer) in the pretrained file (segformer.b5.640x640.ade.160k.pth)
```
mix_transformer_ourprompt_proj.py

pretrained_dict = torch.load("/home/zjz/code/DMPNet/pretrained/segformer.b5.640x640.ade.160k.pth")  Your Path
```

### Testing
```
cd ./RGBD
python evaluate.py --logdir "MODEL PATH"
```


## Acknowledgements

This repo is based on [DPLNet](https://github.com/ShaohuaDong2021/DPLNet)  which is an excellent work.
 
