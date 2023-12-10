### Implementation of CycleGAN for RGB-to-Tactile map translation

##### Installation setup
```shell
conda create -f conda_env.yaml
```

##### Data directory structure:
.
└── data/
    ├── train/
    │   ├── rgb
    │   └── tactile
    └── val/
        ├── rgb
        └── tactile


##### Training the model

Modify the Trainer instance in the src/train.py file:
```py
trainer = Trainer(
    img_size=(256, 256),
    epochs=100,
    batch_size=20,
    shuffle=True,
    train_D_every=5,
    save_model_every=4,
    save_samples_every=4,
    save_model_dir='checkpoints',
    save_samples_dir='samples',
    # resume_ckpt_dir='checkpoints/91'
  )

  l_G, l_C, l_IDT, l_D = trainer.train()
```

Run train.py as a python module:
```shell
python -m src.train
```

##### FID metric
Provide the path to the netG_AB generator in the main function of the src/metrics/fid.py

```py
...
netG_AB.load_state_dict(torch.load('checkpoints/81/netG_AB.pt'))
...
```
Run the fid.py file as a python module
```shell
python -m src.metrics.fid
```

##### LPIPS metric
Provide the path to the netG_AB generator in the main function of the src/metrics/lpips.py

```py
...
netG_AB.load_state_dict(torch.load('checkpoints/81/netG_AB.pt'))
...
```
Run the fid.py file as a python module
```shell
python -m src.metrics.lpips
```

##### Inference
Modify the src/inference.py file

```py
weight_path = 'checkpoints/81/netG_AB.pt'
_, loader = load_data('data', img_size=(256, 256))
output_path = 'inference'
inference(weight_path, output_path, loader)
...
```
Run the inference.py file as a python module:
```shell
python -m src.inference
```
