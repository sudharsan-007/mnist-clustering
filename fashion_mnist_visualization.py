
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms
import pytorch_lightning as pl

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class LitAutoEncoder(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
								nn.Linear(28 * 28, 256),
								nn.SELU(),
								nn.Linear(256, 64),
								nn.SELU(),
								nn.Linear(64, 10))
		self.decoder = nn.Sequential(
								nn.Linear(10, 64),
								nn.SELU(),
								nn.Linear(64, 256),
								nn.SELU(),
								nn.Linear(256, 28 * 28))

	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)    
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)
        
	def img_reconstruction(self, img):
		encoded_x = self.encoder.forward(img.reshape(-1))
		decoded_x = self.decoder.forward(encoded_x)
		recon_img = decoded_x.detach().numpy()
		recon_img = img.reshape(28,28,1)
		fig = plt.figure()
		ax1 = fig.add_subplot(1,2,1)
		ax1.imshow(img.reshape(28,28,1))
		ax2 = fig.add_subplot(2,2,2)
		ax2.imshow(recon_img)
		plt.show()
    
def plot_results(tsne_2d,x,y):
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.tab10
    plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y, s=10, cmap=cmap)
    image_positions = np.array([[1., 1.]])
    for index, position in enumerate(tsne_2d):
        dist = np.sum((position - image_positions) ** 2, axis=1)
        if np.min(dist) > 100: # if far enough from other images
            image_positions = np.r_[image_positions, [position]]
            imagebox = mpl.offsetbox.AnnotationBbox(
                mpl.offsetbox.OffsetImage(x[index].reshape(28,28), cmap="binary"),
                position, bboxprops={"edgecolor": cmap(y[index]), "lw": 2})
            plt.gca().add_artist(imagebox)
    plt.axis("off")
    # save_fig("fashion_mnist_visualization_plot")
    plt.show() 


if __name__ == "__main__":
    
    dataset = FashionMNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(mnist_train, batch_size=32,num_workers=4)
    val_loader = DataLoader(mnist_val, batch_size=32,num_workers=4)
    model = LitAutoEncoder()
    
    # training
    #trainer = pl.Trainer(gpus=1, num_nodes=8, precision=16, limit_train_batches=0.5)
    trainer = pl.Trainer(
		max_epochs=30,
		accelerator="auto",
		precision=16,
		devices = 1,  # no of GPU change to 1 for single gpu
		logger=CSVLogger(save_dir="logs/"),
		callbacks=[TQDMProgressBar(refresh_rate=5)]
		)
    
    trainer.fit(model, train_loader, val_loader)
    val_data = DataLoader(mnist_val, batch_size=5000,num_workers=4)
    f, l = next(iter(val_data))
    encoder_predictions = model.forward(f.view(f.size(0), -1))
    tsne = TSNE()
    tsne_2d = tsne.fit_transform(encoder_predictions.detach().numpy())
    plot_results(tsne_2d,f,l)
    
    model.img_reconstruction(dataset[0][0])



