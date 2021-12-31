# 添加motion_disc_loader
from torch.utils.data import ConcatDataset, DataLoader
from .amass import AMASS

def CreateDataLoader(opt,cfg):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)

    # ===== Motion Discriminator dataset =====
    motion_disc_db = AMASS(seqlen=cfg.DATASET.SEQLEN)

    motion_disc_loader = DataLoader(
        dataset=motion_disc_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    return data_loader, motion_disc_loader

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