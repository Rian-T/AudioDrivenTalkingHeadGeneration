import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.model_utils as model_utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import mean_absolute_percentage_error

class AudioExpressionNet3(pl.LightningModule):
    def __init__(self, T=8, test_init=True, learning_rate=0.0001):
        super(AudioExpressionNet3, self).__init__()
        
        def _set_requires_grad_false(layer):
            for param in layer.parameters():
                param.requires_grad = False

        #self.expression_dim = 4 * 512
        self.expression_dim = 20
        self.T = T

        self.learning_rate = learning_rate

        self.convNet = nn.Sequential(
            nn.Conv1d(29, 32, 3, stride=2, padding=1),  # [b, 32, 8]
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),  # [b, 32, 4]
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),  # [b, 64, 2]
            nn.LeakyReLU(0.02),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),  # [b, 64, 1]
            nn.LeakyReLU(0.02),
        )

        # # Load pre-trained convNet
        # if not test_init:
        #     self.convNet.load_state_dict(torch.load(
        #         'model/audio2expression_convNet_justus.pt'))

        # latent_dim = 128
        #pca_dim = 512
        # self.latent_in = nn.Linear(self.expression_dim, latent_dim)

        # # Initialize latent_in with pca components
        # if not test_init:
        #     pca = 'model/audio_dataset_pca512.pt'
        #     weight = torch.load(pca)[:latent_dim]
        #     with torch.no_grad():
        #         self.latent_in.weight = nn.Parameter(weight)

        self.fc1 = nn.Linear(64, 128)
        #self.adain1 = model_utils.LinearAdaIN(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        #self.fc_out = nn.Linear(pca_dim, self.expression_dim)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 20)

        # # Init fc_out with 512 precomputed pca components
        # if not test_init:
        #     pca = 'model/audio_dataset_offset_to_mean_4to8_pca512.pt'
        #     weight = torch.load(pca)[:pca_dim].T
        #     with torch.no_grad():
        #         self.fc_out.weight = nn.Parameter(weight)

        # attention
        self.attentionNet = nn.Sequential(
            # b x expression_dim x T => b x 256 x T
            nn.Conv1d(self.expression_dim, 256, 3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 256 x T => b x 64 x T
            nn.Conv1d(256, 64, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 64 x T => b x 16 x T
            nn.Conv1d(64, 16, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 16 x T => b x 4 x T
            nn.Conv1d(16, 4, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 4 x T => b x 1 x T
            nn.Conv1d(4, 1, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Flatten(),
            nn.Linear(self.T, self.T, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, audio):
        # input shape: [b, T, 16, 29]
        #print(audio.shape)
        b = audio.shape[0]
        audio = audio.permute(0, 1, 3, 2)  # [b, T, 29, 16]
        audio = audio.view(b * self.T, 29, 16)  # [b * T, 29, 16]

        #print("Audio Encoder Input Shape : " + str(audio.shape))
        # Convolution
        conv_res = self.convNet(audio)
        conv_res = conv_res.view(b * self.T, 1, -1)  # [b * T, 1, 64]

        #print("Audio Encoder Output Shape : " + str(conv_res.shape))

        #latent = self.latent_in(latent.clone().view(b, -1))

        # Fully connected
        expression = []
        conv_res = conv_res.view(b, self.T, 1, -1)  # [b, T, 1, 64]
        conv_res = conv_res.transpose(0, 1)  # [T, b, 1, 64]
        for t in conv_res:
            #z_ = F.leaky_relu(self.adain1(self.fc1(t), latent), 0.02) #original line
            z_ = F.leaky_relu(self.fc1(t))
            z_ = F.leaky_relu(self.fc2(z_))
            z_ = F.leaky_relu(self.fc3(z_))
            z_ = F.leaky_relu(self.fc4(z_))
            z_ = F.leaky_relu(self.fc5(z_))
            z_ = F.leaky_relu(self.fc6(z_))
            expression.append(self.fc_out(z_))
        expression = torch.stack(expression, dim=1)  # [b, T, expression_dim]

        expression = expression.squeeze(2)
        #print("Mapping network output Shape : " + str(expression.shape))

        #return expression
        
        # expression = expression[:, (self.T // 2):(self.T // 2) + 1]

        if self.T > 1:
            expression_T = expression.transpose(1, 2)  # [b, expression_dim, T]
            attention = self.attentionNet(
                expression_T).unsqueeze(-1)  # [b, T, 1]
            expression = torch.bmm(expression_T, attention)

        #print("Attention output Shape : " + str(expression.view(b, 4, 512).shape))
        return expression.view(b, 1, 20)  # shape: [b, 4, 512]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        audio, A = batch
        A_hat = self.forward(audio)
        loss = F.mse_loss(A_hat, A)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        #self.log('train_mape_step', mean_absolute_percentage_error(A_hat, A))
        self.lr_schedulers()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.4, verbose=True)
        
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_loss passed in as checkpoint_on
            'monitor': 'valid_loss'
        }
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        audio, A = batch
        A_hat = self.forward(audio)
        loss = F.mse_loss(A_hat, A)
        self.log("valid_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #self.log('valid_mape_step', mean_absolute_percentage_error(A_hat, A))
        return loss