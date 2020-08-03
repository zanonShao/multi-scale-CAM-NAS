import pytorch_lightning as pl
from trapezoid_supernet import trapezoid_supernet
model = trapezoid_supernet(max_scale=4, num_layers=12)
# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(model)