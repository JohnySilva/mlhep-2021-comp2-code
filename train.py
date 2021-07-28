import configparser
import pathlib as path

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from idao.data_module import IDAODataModule
from idao.model import SimpleConv, Print
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
#device="cuda:2"
#%%

def trainer(mode: ["classification", "regression"], cfg, dataset_dm, filename):
    model = SimpleConv(mode=mode)#.to(device)
    if mode == "classification":
        epochs = cfg["TRAINING"]["ClassificationEpochs"]
    else:
        epochs = cfg["TRAINING"]["RegressionEpochs"]
 
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoint_model/',
    filename=filename,
    save_top_k=3,
    mode='min',
    )
    
    trainer = pl.Trainer(
        gpus=int(cfg["TRAINING"]["NumGPUs"]),
        max_epochs=int(epochs),
        progress_bar_refresh_rate=20,
        weights_save_path=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]).joinpath(
            mode
        ),
        default_root_dir=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]),
        callbacks=[EarlyStopping(monitor="val_loss"), checkpoint_callback]
    )

    # Train the model ⚡
    trainer.fit(model, dataset_dm)
#%%    
def main():
    seed_everything(666)
    config = configparser.ConfigParser()
    config.read("./config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    dataset_dm.setup()
    filename='best_model_52/best_model-{epoch:02d}-{val_loss:.2f}'

    #for mode in ["classification", "regression"]:
    mode = "regression"
    print(f"Training for {mode}")
    trainer(mode, cfg=config, dataset_dm=dataset_dm, filename=filename)

if __name__ == "__main__":
    main()
#%%