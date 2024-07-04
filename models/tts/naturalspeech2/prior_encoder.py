# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.naturalpseech2.transformers import (
    TransformerEncoder,
    DurationPredictor,
    PitchPredictor,
    LengthRegulator,
)


class PriorEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc_emb_tokens = nn.Embedding(
            cfg.vocab_size, cfg.encoder.encoder_hidden, padding_idx=0
        )
        self.enc_emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)
        self.encoder = TransformerEncoder(
            enc_emb_tokens=self.enc_emb_tokens, cfg=cfg.encoder
        )

        self.duration_predictor = DurationPredictor(cfg.duration_predictor)
        self.pitch_predictor = PitchPredictor(cfg.pitch_predictor)
        self.length_regulator = LengthRegulator()

        self.pitch_min = cfg.pitch_min
        self.pitch_max = cfg.pitch_max
        self.pitch_bins_num = cfg.pitch_bins_num

        pitch_bins = torch.exp(
            torch.linspace(
                np.log(self.pitch_min), np.log(self.pitch_max), self.pitch_bins_num - 1
            )
        )
        self.register_buffer("pitch_bins", pitch_bins)

        self.pitch_embedding = nn.Embedding(
            self.pitch_bins_num, cfg.encoder.encoder_hidden
        )

    def forward(
        self,
        phone_id,  # [1,8]
        duration=None,
        pitch=None,
        phone_mask=None,
        mask=None,
        ref_emb=None,  # [1,512,670]
        ref_mask=None,  # [1,670]
        is_inference=False,  # True
    ):
        """
        input:
        phone_id: (B, N)
        duration: (B, N)
        pitch: (B, T)
        phone_mask: (B, N); mask is 0
        mask: (B, T); mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'); mask is 0

        output:
        prior_embedding: (B, d, T)
        pred_dur: (B, N)
        pred_pitch: (B, T)
        """

        x = self.encoder(phone_id, phone_mask, ref_emb.transpose(1, 2))  # TransformerEncoder [1,8,512]
        # print(torch.min(x), torch.max(x))
        dur_pred_out = self.duration_predictor(x, phone_mask, ref_emb, ref_mask) # phone_mask None  ref_emb [1,512,670] ref_mask [1,670]
        # dur_pred_out: {dur_pred_log, dur_pred, dur_pred_round}

        if is_inference or duration is None:
            x, mel_len = self.length_regulator(
                x,
                dur_pred_out["dur_pred_round"],
                max_len=torch.max(torch.sum(dur_pred_out["dur_pred_round"], dim=1)),
            )
        else:
            x, mel_len = self.length_regulator(x, duration, max_len=pitch.shape[1])
        # x [1,60,512] mel_len [60]
        pitch_pred_log = self.pitch_predictor(x, mask, ref_emb, ref_mask)
        # pitch_pred_log [1,60]
        if is_inference or pitch is None:  # self.pitch_bins   [50,1100]  len(self.pitch_bins) = 511
            pitch_tokens = torch.bucketize(pitch_pred_log.exp(), self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)  # pitch_embedding [1,60,512]
        else:
            pitch_tokens = torch.bucketize(pitch, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)

        x = x + pitch_embedding

        if (not is_inference) and (mask is not None):
            x = x * mask.to(x.dtype)[:, :, None]

        prior_out = {
            "dur_pred_round": dur_pred_out["dur_pred_round"],
            "dur_pred_log": dur_pred_out["dur_pred_log"],
            "dur_pred": dur_pred_out["dur_pred"],
            "pitch_pred_log": pitch_pred_log,
            "pitch_token": pitch_tokens,
            "mel_len": mel_len,
            "prior_out": x,
        }

        return prior_out
