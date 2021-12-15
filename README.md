# RNN-MBP
Deep Recurrent Neural Network with Multi-scale Bi-directional Propagation for Video Deblurring (AAAI-2022) 
by [Chao Zhu](https://www.sivlab-xjtu.com/), [Hang Dong](https://sites.google.com/view/hdong/%E9%A6%96%E9%A1%B5), [Jinshan Pan](https://jspan.github.io/), Boyang Liang, Yuhao Huang, Lean Fu, and [Fei Wang](https://www.sivlab-xjtu.com) 

[[Paper]](https://arxiv.org/abs/2112.05150) [[Supp]](https://drive.google.com/drive/folders/1i0EdcaSnSIrn38jm6nwKtANacTct083R?usp=sharing)


## Results

### Results on GOPRO
![image](https://github.com/XJTU-CVLAB-LOWLEVEL/RNN-MBP/blob/main/example_results/GORPO/GOPRO.png)

### Results on DVD
![image](https://github.com/XJTU-CVLAB-LOWLEVEL/RNN-MBP/blob/main/example_results/DVD/DVD.png)

### Results on RBVD
![image](https://github.com/XJTU-CVLAB-LOWLEVEL/RNN-MBP/blob/main/example_results/RBVD/RBVD.png)


## Prerequisites

- Python 3.6 
- PyTorch 1.8
- opencv-python
- scikit-image
- lmdb
- thop
- tqdm
- tensorboard



## Real-world Bluryy Video Dataset (RBVD)
We have collected a new [RBVD dataset](https://drive.google.com/drive/folders/1YQUIGdW4SCAQW5-dxg2lwjTig2XKLeSG?usp=sharing) with more scenes and perfect alignment, using the proposed Digital Video Acquisition System.



## Training
Please download and unzip the dataset file for each benchmark.

- [GOPRO](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
- [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)
- [RBVD](https://drive.google.com/drive/folders/1YQUIGdW4SCAQW5-dxg2lwjTig2XKLeSG?usp=sharing)

Then, specify the *\<path\>* (para.data_root) where you put the dataset file and the corresponding dataset configurations in the command (e.g. para.dataset=gopro or gopro_ds_lmdb).

The default training process requires at least 4 NVIDIA Tesla V100 32Gb GPUs.

The training command is shown below:

```bash
python main.py --data_root <path> --dataset gopro_ds_lmdb  --num_gpus 4 --batch_size 4  --patch_size [256, 256]  --end_epoch 500
```


## Testing
Please download [checkpoints](https://drive.google.com/drive/folders/1i0EdcaSnSIrn38jm6nwKtANacTct083R?usp=sharing) and unzip it under the Source directory.

Example command to run a pre-trained model:

```bash
python test.py --data_root <path> --dataset gopro_ds_lmdb  --test_only --test_checkpoint <path>  --model RNN-MBP 
```


## Citing

If you use any part of our code, or RNN-MBP and RBVD are useful for your research, please consider citing:

```bibtex
@inproceedings{chao2022,
  title={Deep Recurrent Neural Network with Multi-scale Bi-directional Propagation for Video Deblurring},
  author={Chao, Zhu and Hang, Dong and Jinshan, Pan and Boyang, Liang and Yuhao, Huang and Lean, Fu and Fei, Wang},
  booktitle={AAAI},
  year={2022},
}
```
