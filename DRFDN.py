import torch
import torch.nn as nn
import mkdc_block as M


class DRFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(DRFDN, self).__init__()

        self.fea_conv = M.conv_layer(in_nc, nf, kernel_size=5)


        self.B1 = M.DRFDB(in_channels=nf, mid_channel = mf)
        self.B2 = M.DRFDB(in_channels=nf, mid_channel = mf)
        self.B3 = M.DRFDB(in_channels=nf, mid_channel = mf)
        self.B4 = M.DRFDB(in_channels=nf, mid_channel = mf)

        self.LR_conv = M.conv_layer(nf, nf, kernel_size=5)

        upsample_block = M.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0



    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
