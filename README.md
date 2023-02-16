# WiCoNet
Pytorch codes of 'Looking Outside the Window: Wider Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images' [[paper]](https://doi.org/10.1109/TGRS.2022.3168697)

**BLU dataset** [[download link]](https://rslab.disi.unitn.it/dataset/BLU/) [[Baidu Netdisk]](https://pan.baidu.com/s/117F0c5eR56Y97cQDIXD1Sg?pwd=uf7c)

![alt text](https://github.com/ggsDing/WiCoNet/blob/main/WiCoNet.png)
![alt text](https://github.com/ggsDing/WiCoNet/blob/main/data_BLU.png)

**To be updated:**
- [x] Codes for the BLU dataset
- [x] Codes for the GID
- [x] Codes for the Potsdam dataset
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

If you find our work useful or interesting, please consider to cite:
> L. Ding et al., "Looking Outside the Window: Wide-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3168697.
