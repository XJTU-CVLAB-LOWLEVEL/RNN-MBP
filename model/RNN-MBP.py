import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import *
# from model import flow_pwc
# from model.flow_pwc import Backward

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

## Original Resolution Block (ORB)
class CABs(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(CABs, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# RDB-based RNN cell
class shallow_cell(nn.Module):
    def __init__(self, para):
        super(shallow_cell, self).__init__()
        self.n_feats = para.n_features
        act = nn.PReLU()
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(3, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat


class Encoder(nn.Module):
    def __init__(self, para, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Encoder, self).__init__()
        n_feat = para.n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # U-net skip
        self.skip_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                   bias=bias)

        self.skip_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                   bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.skip_enc1(encoder_outs[0]) + self.skip_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.skip_enc2(encoder_outs[1]) + self.skip_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.skip_enc3(encoder_outs[2]) + self.skip_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, para, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Decoder, self).__init__()
        n_feat = para.n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
class TFR(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(TFR, self).__init__()

        self.orb1 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class Model(nn.Module):

    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda')

        # RNN cell
        self.shallow_cell = shallow_cell(para)
        self.forward_encoder = Encoder(self.para)
        self.forward_decoder = Decoder(self.para)
        self.backward_encoder = Encoder(self.para)
        self.backward_decoder = Decoder(self.para)

        # PWC
        # self.flow_net = flow_pwc.Flow_PWC()

        # Fusion
        self.TFR = TFR(n_feat=self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=int(self.n_feats/2), num_cab=8)
        self.fusion =  conv3x3(3 * self.n_feats, self.n_feats)
        self.recons = conv5x5(self.n_feats, 3)


    def forward(self, x):

        batch_size, frames, channels, height, width = x.shape

        # feature extractor cell
        features = []
        for i in range(frames):
            feature = self.shallow_cell(x[:, i, :, :, :])
            features.append(feature)

        ##################
        #      MBP
        ##################
        # forward cell
        encoder_outs = None
        decoder_outs = None
        encoder_outs_list = []
        decoder_outs_list = []
        for i in range(frames):
            encoder_outs = self.forward_encoder(features[i], encoder_outs, decoder_outs)
            decoder_outs = self.forward_decoder(encoder_outs)
            encoder_outs_list.append(encoder_outs)
            decoder_outs_list.append(decoder_outs)

        # backward cell
        encoder_outs = None
        decoder_outs = None
        encoder_outs_list_2 = []
        decoder_outs_list_2 = []
        for i in range(frames):
            encoder_outs = self.backward_encoder(features[frames-i-1], encoder_outs, decoder_outs)
            decoder_outs = self.backward_decoder(encoder_outs)
            encoder_outs_list_2.append(encoder_outs)
            decoder_outs_list_2.append(decoder_outs)
        encoder_outs_list_2 = list(reversed(encoder_outs_list_2))
        decoder_outs_list_2 = list(reversed(decoder_outs_list_2))

        # concat forward and backward
        for i in range(frames):
            for j in range(3):
                encoder_outs_list[i][j] = encoder_outs_list[i][j]+encoder_outs_list_2[i][j]
                decoder_outs_list[i][j] = decoder_outs_list[i][j]+decoder_outs_list_2[i][j]


        ##################
        #       TFR
        ##################
        outputs = []
        for i in range(self.num_fb, frames - self.num_ff):
            # input_features = self.fusion(features[i])
            output_features = self.TFR(features[i], encoder_outs_list[i], decoder_outs_list[i])
            deblurred_img = self.recons(output_features)
            deblurred_img = x[:, i, :, :, :] +deblurred_img
            outputs.append(deblurred_img.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)



def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params
