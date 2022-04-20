# WiCoNet
Pytorch codes of 'Looking Outside the Window: Wider Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images' [[paper]](https://doi.org/10.1109/TGRS.2022.3168697)

**BLU dataset** [[download link]](https://rslab.disi.unitn.it/dataset/BLU/)

![alt text](https://github.com/ggsDing/WiCoNet/blob/main/WiCoNet.png)
![alt text](https://github.com/ggsDing/WiCoNet/blob/main/data_BLU.png)

**To be updated:**
- [x] Training codes to the BLU dataset
- [ ] Training codes to the GID
- [ ] Training codes to the Potsdam dataset
- [ ] Optimizing the codes to easily switch datasets

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
