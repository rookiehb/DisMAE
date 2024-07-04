# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
import torch.autograd as autograd
from torch.autograd import Variable


class MaskedAutoencoderViT(nn.Module):
    """ 
        Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, attr_depth=2, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, args=None, norm_pix_loss=False):
        super().__init__()

        self.in_chans = in_chans
        self.args = args

        # DomainNet & Unsupervised Domain Generalization setting     
        self.num_classes = 20

        self.softmax = nn.Softmax(dim=-1)
        self.reweight_classifier = nn.Sequential(
                                    nn.Linear(embed_dim, embed_dim*3),
                                    nn.ReLU(), nn.Linear(embed_dim*3,3)
                                    )
        
        self.d_classifier = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim*3), nn.ReLU(), 
                            nn.Linear(embed_dim*3, embed_dim*3), nn.ReLU(),
                            nn.Linear(embed_dim*3, 3)
                            )

        if self.args.is_linear:
            self.s_classifier = nn.Sequential(nn.Linear(embed_dim, self.num_classes))
        else:
            self.s_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//2), nn.ReLU(),
                nn.Linear(embed_dim//2, embed_dim//4), nn.ReLU(),
                nn.Linear(embed_dim//4, self.num_classes), nn.Sigmoid()
            )
        
        self.criterion = nn.CrossEntropyLoss()

        # --------------------------------------------------------------------------
        # Disentangle MAE loss function
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.variation_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(attr_depth)])

        self.norm = norm_layer(embed_dim)
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.sigmoid = nn.Sigmoid()
        
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        self.domain_labels = torch.zeros([args.batch_size*3], dtype=torch.int64).cuda()
        self.domain_labels[args.batch_size:args.batch_size*2] = 1
        self.domain_labels[args.batch_size*2:args.batch_size*3] = 2
        self.domain_labels = Variable(self.domain_labels)

    def frozen_backbone(self):
        
        for k, v in self.blocks.named_parameters():
            v.requires_grad_(False)

        for k, v in self.variation_blocks.named_parameters():
            v.requires_grad_(False)

        for k, v in self.decoder_blocks.named_parameters():
            v.requires_grad_(False)
            
        for k, v in self.reweight_classifier.named_parameters():
            v.requires_grad_(True)

    def frozen_cls(self):
        for k, v in self.blocks.named_parameters():
            v.requires_grad_(True)

        for k, v in self.variation_blocks.named_parameters():
            v.requires_grad_(True)

        for k, v in self.decoder_blocks.named_parameters():
            v.requires_grad_(True)

        for k, v in self.reweight_classifier.named_parameters():
            v.requires_grad_(False)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        # 根据网络层的不同定义不同的初始化方式
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
            imgs: (N, 3, H, W)
            x: (N, L, patch_size**2 *3)
        """
        # print("input imgs shape ", imgs.shape)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
            x: (N, L, patch_size**2*3)
            imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        # embed patches    
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if not self.training and not self.args.recon_mission:
            x_s, mask_s, ids_restore_s = self.random_masking(x, 0.0)
        else:
            x_s, mask_s, ids_restore_s = self.random_masking(x, mask_ratio)

        v_mask_ratio = mask_ratio-0.10
        x_v, mask_v, ids_restore_v = self.random_masking(x, v_mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x_s = torch.cat((cls_tokens, x_s), dim=1)
        x_v = torch.cat((cls_tokens, x_v), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x_s = blk(x_s)

        for blk in self.variation_blocks:
            x_v = blk(x_v)

        x_s = self.norm(x_s)
        x_v = self.norm(x_v)

        return x_s, mask_s, ids_restore_s, x_v, mask_v, ids_restore_v

    def forward_decoder(self, x):

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        x = x[:, 1:-1, :]
        
        x = self.sigmoid(x)

        return x

    def recon_loss(self, imgs, pred, mask):
        
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        
        loss = (pred - target) ** 2
        
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss*mask).sum() / (mask.sum()+1)
        return loss

    def ada_loss(self, imgs, pred, mask, index):
        """
            imgs: [N, 3, H, W]
            pred: [N, L, p*p*in_chans]
            mask: [N, L], 0 is keep, 1 is remove,
        """
        imgs = torch.unsqueeze(imgs, dim=0)
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = torch.max(loss-self.args.margin, torch.zeros_like(loss))
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.sum(dim=-1) / (mask.sum(dim=-1)+1)

        pos_loss = torch.sum(torch.exp(-loss[index]/self.args.tao))
        neg_loss = torch.sum(torch.exp(-loss/self.args.tao))
        ada_loss = -torch.log(pos_loss/neg_loss)

        return ada_loss

    def unmask(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        return x

    def pred_domain(self, imgs):
        with torch.no_grad():
            latent_s, mask_s, ids_restore_s, latent_v, mask_v, ids_restore_v = self.forward_encoder(imgs, 0.15)
            latent_s = latent_s.detach()
            latent_v = latent_v.detach()

            s_cls2reweight = latent_s[:, 0, :]
            v_cls = latent_v[:, 0, :]

        return self.reweight_classifier(s_cls2reweight)
        

    def forward(self, imgs, labels, mask_ratio=0.5):
        latent_s, mask_s, ids_restore_s, latent_v, mask_v, ids_restore_v = self.forward_encoder(imgs, mask_ratio)
        
        s_cls = latent_s[:, 0, :]
        predict_label = self.s_classifier(s_cls)
        pred_loss = self.criterion(predict_label, labels)
        
        latent_s = self.unmask(latent_s, ids_restore_s) # shape [batch_size*5, patch**2+1, decoder_embed_dim]
        latent_v = self.unmask(latent_v, ids_restore_v)
        
        ada_loss = 0
        batch_size = imgs.shape[0]
        v_cls = latent_v[:, 0, :].unsqueeze(1)
        con_feat = torch.cat((latent_s, v_cls), dim=1)

        recon_img = self.forward_decoder(con_feat)
        domain_size = int(batch_size//3)
        
        if self.training:

            s_cls2reweight = s_cls.detach()
            predict_domain = self.reweight_classifier(s_cls2reweight)
            pred_domain_loss = self.criterion(predict_domain, self.domain_labels)

            recon_loss = self.recon_loss(imgs, recon_img, mask_s)
            
            for i in range(self.args.ada_iter, -1, -1):
                for bid in range(3):
                    index_d = bid*domain_size+i
                    repeat_s_d = latent_s[index_d].repeat(domain_size, 1)
                    repeat_s_d = repeat_s_d.reshape(domain_size, latent_s.shape[1], latent_s.shape[2])

                    con_v_cls = v_cls[index_d:index_d+domain_size]

                    switch_feat_abla = torch.cat((repeat_s_d, con_v_cls), dim=1)
                    abla_img = self.forward_decoder(switch_feat_abla)
                    repeat_mask_s = mask_s[index_d, :].repeat(domain_size, 1)
                    assert repeat_mask_s.shape[0] == domain_size

                    softmax_predict_domain = self.softmax(predict_domain)
                    gt_weights = softmax_predict_domain[index_d:index_d+self.args.batch_size, bid].detach()
                    
                    gt_weight = max(0.15, gt_weights[0])
                    ada_loss += 1/gt_weight*self.ada_loss(imgs[index_d], abla_img, repeat_mask_s, index_d-bid*domain_size)

            ada_loss /= self.args.ada_iter+1
            loss = recon_loss + self.args.lambda1*ada_loss+ self.args.lambda2*pred_loss
            
            loss_status = {'recon_loss':recon_loss, 'ada_loss':ada_loss, 'pred_loss':pred_loss,
                            'total_loss':loss, "pred_domain_loss": pred_domain_loss}
            
            return loss_status, predict_label, softmax_predict_domain

def mae_vit_recon(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, attr_depth=6, 
        num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_cls(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, attr_depth=6, 
        num_heads=12, decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=12, attr_depth=6, 
        num_heads=3, decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
