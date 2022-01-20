# 添加motion_disc_loader
from torch.utils.data import ConcatDataset, DataLoader
from .amass import AMASS

def CreateDataLoader(opt,cfg):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)

    # ===== Motion Discriminator dataset =====
    if opt.use_pixmaf:  
        motion_disc_db = AMASS(seqlen=cfg.DATASET.SEQLEN)

        motion_disc_loader = DataLoader(
            dataset=motion_disc_db,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads),
        )
        return data_loader, motion_disc_loader
    
    else:
        return data_loader

def CreateDataLoaderU(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoaderU
    data_loader = CustomDatasetDataLoaderU()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateTransferDataLoader(opt):
    from data.custom_dataset_data_loader import CustomTransferDatasetDataLoader
    data_loader = CustomTransferDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader