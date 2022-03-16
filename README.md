# WiCoNet
Pytorch codes of 'Looking Outside the Window: Wider Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images' [[paper]](http://arxiv.org/abs/2106.15754)

![alt text](https://github.com/ggsDing/WiCoNet/blob/main/flow_chart.png)
BLU dataset [[download link]](https://rslab.disi.unitn.it/dataset/BLU/)
![alt text](https://github.com/ggsDing/WiCoNet/blob/main/data_BLU.png)

**How to Use**
1. Split the data into training, validation and test set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - image
>    - label
>  - Val
>    - image
>    - label
>  - Test
>    - image
>    - label

2. Change the training parameters in *Train_WiCo_BLU.py*, especially the data directory.

3. To evaluate, change also the parameters in *Eval_WiCo_BLU.py*, especially the data directory and the checkpoint path.
