from para import Parameter
from train import Trainer

if __name__ == '__main__':
    para = Parameter().args

    ##################
    #  Para settings
    ##################
    # The following Para settings are used for fast training.
    # For more details of parameters, please refer ./para/parameter.py

    # # train dataset
    # para.dataset = 'BSD'
    para.dataset = 'gopro_ds_lmdb'
    # para.dataset = 'gopro'
    # para.dataset = 'reds_lmdb'
    # para.dataset = 'BSD'
    # para.dataset = 'DeepVideoDeblurring_lmdb'

    # # dataset root
    para.data_root = 'E:/Datasets'

    # # resume training from existing checkpoint
    # para.resume = True
    # para.resume_file = './experiment/2020_12_29_01_56_43_RNN-MBP_gopro_ds_lmdb/model_best.pth.tar'

    # train settings
    para.num_gpus = 4
    para.batch_size = 4
    para.patch_size = [256, 256]
    para.end_epoch = 500


    ##################
    #  main process
    ##################
    para.model = 'RNN-MBP'

    trainer = Trainer(para)
    trainer.run()
