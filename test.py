from para import Parameter
from train import Trainer

if __name__ == '__main__':
    para = Parameter().args

    # test parameters
    para.test_only = True
    para.test_save_dir = './results/'
    para.test_frames = 20

    # test model
    para.model='RNN-MBP'
    para.test_checkpoint = './experiment/2020_12_29_01_56_43_ESTRNN_gopro_ds_lmdb/model_best.pth.tar'

    # test dataset
    para.data_root = 'E:/Datasets'
    para.dataset = 'gopro_ds_lmdb'   # gopro_ds_lmdb & rbvd_lmdb & reds_lmdb  DeepVideoDeblurring_lmdb

    trainer = Trainer(para)
    trainer.run()
