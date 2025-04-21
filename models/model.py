
import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop
from models.schedule import LinearWarmupCosineAnnealingLR
from models.loss import ProtoLearningLoss, ProtoGuidedLoss
import re
from models.resnet import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParaGuide(nn.Module):
    def __init__(self, args):
        super(ParaGuide, self).__init__()
        self.args = args
        self.encoder = RN18_Encoder(pretrained=False)
        self.pro_fea = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        
        self.auxiliary_encoder = RN18_Encoder(pretrained=False)
        self.pro_aux = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        
        # Parameters for encoders and projection layers
        self.enc_params = self.encoder.parameters()
        self.pro_params = self.pro_fea.parameters()
        self.aux_enc_params = self.auxiliary_encoder.parameters()
        self.pro_aux_params = self.pro_aux.parameters()

        # Placeholders for optimizer, scheduler, and loss functions
        self.optimizer = None
        self.scheduler = None
        self.criterion_aux = None
        self.criterion_guidance = None
    
    def initialize_training_components(self):
        """Initialize optimizer, scheduler, and loss functions for training."""
        self.criterion_aux = ProtoLearningLoss(contrast_mode='all')
        self.criterion_guidance = ProtoGuidedLoss(contrast_mode='one')

        self.optimizer = RMSprop([
            {"params": self.enc_params, "lr": self.args.learning_rate},
            {"params": self.pro_params, "lr": self.args.learning_rate},
            {"params": self.aux_enc_params, "lr": self.args.learning_rate},
            {"params": self.pro_aux_params, "lr": self.args.learning_rate}
        ], self.args.learning_rate, weight_decay=0.0005)

        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, self.args.warmup_epochs, self.args.max_epochs
        )

    def train_iter(self, batch):
        """Perform a single training iteration."""
        # Handwriting Prototype Learning
        pro_emb = self.pro_aux(self.auxiliary_encoder(batch['prototype_img'].to(device), pool=True))
        pro_pos = self.pro_aux(self.auxiliary_encoder(batch['contrast_img'].to(device), pool=True))
        nce_emb_aux = torch.stack([pro_emb, pro_pos], dim=1) #[B , 2, C]
        nce_emb_aux = nn.functional.normalize(nce_emb_aux, p=2, dim=2)
        nce_loss_aux = self.criterion_aux(nce_emb_aux, batch['writer_id'].to(device))

        # Prototype-guided Feature Learning
        pro_query = self.pro_fea(self.encoder(batch['image'].to(device), pool=True))
        nce_emb_guidance = torch.stack([pro_emb, pro_query], dim=1) #[B, 2, C]
        nce_emb_guidance = nn.functional.normalize(nce_emb_guidance, p=2, dim=2)
        nce_loss_guidance = self.criterion_guidance(nce_emb_guidance, batch['writer_id'].to(device))
        
        nce_loss = nce_loss_aux + nce_loss_guidance
        self.optimizer.zero_grad()
        nce_loss.backward()
        self.optimizer.step()
        return nce_loss_aux.item(), nce_loss_guidance.item()
