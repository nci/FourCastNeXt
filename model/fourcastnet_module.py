import pytorch_lightning as pl
import numpy as np
import math
import sys
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_optimizer import Lamb
from .afnonet import AFNONet
from utils import get_logger

class FourCastNetModule(pl.LightningModule):
  def __init__(
    self,
    means,
    stds,
    base_lr=1e-3,
    grad_accum_schedule=None,
    spatial_size=(640, 1280),
    in_channels=40,
    out_channels=20,
    precision=32,
  ):
    super().__init__()
    self.save_hyperparameters()

    self.out_channels = out_channels

    self.register_buffer('means', means, persistent=True)
    self.register_buffer('stds', stds, persistent=True)

    grid = self.setup_grid()
    self.register_buffer('grid', grid, persistent=False)

    self.net = AFNONet(
      img_size=spatial_size,
      in_chans=in_channels,
      out_chans=out_channels)

  def forward(self, x, net):
    value, flow = net(x)

    x = x[:, -self.out_channels:] #B, [t-1, t], H, W
    B, C, H, W = x.shape
    warp_coords = self.grid.repeat(B*C, 1, 1, 1) + flow.view(B*C, H, W, 2)
    x = x.view(B*C, 1, H, W)
    warped_x = F.grid_sample(x, warp_coords, mode='bilinear', align_corners=True)
    warped_x = warped_x.view(B, C, H, W)
    return warped_x + value

  def setup_grid(self):
    h, w = self.hparams.spatial_size
    xgrid = torch.arange(w)
    xgrid = 2 * xgrid / (w - 1) - 1

    ygrid = torch.arange(h)
    ygrid = 2 * ygrid / (h - 1) - 1
    coords = torch.meshgrid(ygrid, xgrid, indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords.permute(1, 2, 0)

  def preprocess(self, batch, is_training=False):
    new_batch = {}

    rep_dims = (1, batch['input0'].shape[1] // self.means.shape[1], 1, 1)
    if self.hparams.precision == 16:
      dtype = torch.float16
    else:
      dtype = torch.float

    if is_training:
      new_batch['input1'] = (batch['input1'] - self.means.repeat(*rep_dims)) / self.stds.repeat(*rep_dims)
      new_batch['input1'] = new_batch['input1'].to(dtype=dtype)

      for step, target in batch['targets'].items():
        n_pred_steps = step
        target = (target - self.means) / self.stds
        target = target.to(dtype=dtype)
        new_batch['target'] = target
        break

      new_batch['n_pred_steps'] = n_pred_steps

    if (is_training and n_pred_steps > 1) or not is_training:
      new_batch['input0'] = (batch['input0'] - self.means.repeat(*rep_dims)) / self.stds.repeat(*rep_dims)
      new_batch['input0'] = new_batch['input0'].to(dtype=dtype)

    return new_batch

  def get_teacher(self, device):
    model_name = '_teacher_model'
    if model_name not in sys.modules:
      if self.hparams.precision == 16:
        dtype = torch.float16
      else:
        dtype = torch.float

      teacher = AFNONet(
        img_size=self.hparams.spatial_size,
        in_chans=self.hparams.in_channels,
        out_chans=self.hparams.out_channels).to(dtype=dtype, device=device)

      teacher.load_state_dict(self.net.state_dict())
      sys.modules[model_name] = teacher.eval()

    return sys.modules[model_name]

  def training_step(self, batch, batch_idx):
    new_batch = self.preprocess(batch, is_training=True)

    target = new_batch['target']
    n_pred_steps = new_batch['n_pred_steps']

    if n_pred_steps > 1:
      if batch_idx % 2 == 0:
        output1 = self.forward(new_batch['input1'], self.net)
        total_loss = F.mse_loss(output1, target) 
      else:
        input0 = new_batch['input0']
        with torch.inference_mode():
          teacher = self.get_teacher(input0.device)
          for _ in range(n_pred_steps-1):
            output0 = self.forward(input0, teacher)
            input0[:, :self.out_channels, ...] = input0[:, -self.out_channels:, ...]
            input0[:, -self.out_channels:, ...] = output0

        output0 = self.forward(input0, self.net)
        total_loss = F.mse_loss(output0, target) 

    else:
      output1 = self.forward(new_batch['input1'], self.net)
      total_loss = F.mse_loss(output1, target) 

    if self.trainer.is_global_zero and \
      ( batch_idx == 0 or \
        (self.trainer.global_step+1) % self.trainer.log_every_n_steps == 0 or \
        self.trainer.global_step+1 == self.trainer.max_steps - 1 ):
      total_loss_val = total_loss.item()

      with torch.no_grad():
        if n_pred_steps > 1 and batch_idx % 2 != 0:
          rmse_val0 = torch.sqrt(F.mse_loss(output0*self.stds+self.means, target*self.stds+self.means)).item()
        else:
          rmse_val0 = 0.0

        if n_pred_steps == 1 or batch_idx % 2 == 0:
          rmse_val1 = torch.sqrt(F.mse_loss(output1*self.stds+self.means, target*self.stds+self.means)).item()
        else:
          rmse_val1 = 0.0

      msg = f'iter={self.trainer.global_step+1}/{self.trainer.max_steps}, total_loss={total_loss_val:.6f}, n_pred_steps={n_pred_steps}, single-step_rmse={rmse_val1:.6f}, multi-step_rmse={rmse_val0:.6f}'

      logger = get_logger(__name__)
      logger.info(msg)

      if not np.isfinite(total_loss_val):
        raise Exception(f'loss is not finite, loss={total_loss_val}, step={self.trainer.global_step+1}')

    self.schedule_accumulate_grads(self.trainer.global_step)

    return total_loss

  def schedule_accumulate_grads(self, step):
    if self.hparams.grad_accum_schedule is None:
      return

    accumulate_grad_batches = 1
    for sched_step in reversed(self.hparams.grad_accum_schedule.keys()):
      if step >= sched_step:
        accumulate_grad_batches = self.hparams.grad_accum_schedule[sched_step]
        break

    self.trainer.accumulate_grad_batches = accumulate_grad_batches

  def configure_optimizers(self):
    net_params = [p for p in self.parameters() if p.requires_grad]
    optimizer = Lamb(net_params, lr=self.hparams.base_lr, weight_decay=self.hparams.base_lr**2)
    scheduler = CosineAnnealingLR(optimizer, self.trainer.max_steps, eta_min=self.hparams.base_lr*0.1)
    return [optimizer, ], [scheduler, ]

  def predict_step(self, batch, batch_idx):
    assert len(batch['input0'].shape) == 4

    im_height, im_width = batch['input0'].shape[-2:]
    roi_height, roi_width = self.hparams.spatial_size

    out_h_stride = int(im_height / 2.0 + 0.5)
    out_w_stride = int(im_width / 2.0 + 0.5)

    assert out_h_stride <= roi_height
    assert out_w_stride <= roi_width

    n_steps = batch['n_pred_steps']

    preds = torch.zeros(n_steps, self.hparams.out_channels, im_height, im_width, dtype=batch['input0'].dtype)

    for hh in range(0, im_height, roi_height):
      for ww in range(0, im_width, roi_width):
        if hh + roi_height >= im_height:
          in_hh = im_height - roi_height
        else:
          in_hh = hh

        if ww + roi_width >= im_width:
          in_ww = im_width - roi_width
        else:
          in_ww = ww

        roi_input = batch['input0'][..., in_hh:in_hh+roi_height, in_ww:in_ww+roi_width].clone()

        roi_batch = self.preprocess({'input0': roi_input})

        inp = roi_batch['input0'].clone()

        for ii in range(n_steps):
          output = self.forward(inp, self.net)
          inp[:, :self.out_channels, ...] = inp[:, -self.out_channels:, ...]
          inp[:, -self.out_channels:, ...] = output

          unnormalized_out = output * self.stds + self.means
          unnormalized_out = unnormalized_out.to(device='cpu')

          if hh == 0 and ww == 0:
            preds[ii, :, :out_h_stride, :out_w_stride] = unnormalized_out[0, :, :out_h_stride, :out_w_stride]
          elif hh == 0 and ww == roi_width:
            preds[ii, :, :out_h_stride, -out_w_stride:] = unnormalized_out[0, :, :out_h_stride, -out_w_stride:]
          elif hh == roi_height and ww == 0:
            preds[ii, :, -out_h_stride:, :out_w_stride] = unnormalized_out[0, :, -out_h_stride:, :out_w_stride]
          elif hh == roi_height and ww == roi_width:
            preds[ii, :, -out_h_stride:, -out_w_stride:] = unnormalized_out[0, :, -out_h_stride:, -out_w_stride:]

    return preds
