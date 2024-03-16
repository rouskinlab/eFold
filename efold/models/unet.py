import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import init

from ..core.model import Model
from ..core.batch import Batch

from ..config import int2seq

import os, sys

from collections import defaultdict    

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, ".."))

CH_FOLD2 = 1


class U_Net(Model):
    def __init__(self, img_ch=3, output_ch=1, lr: float = 1e-5, optimizer_fn=torch.optim.Adam, **kwargs):

        super().__init__(lr=lr, optimizer_fn=optimizer_fn, **kwargs)
        self.model_type = "Unet"
        self.data_type_output = ["structure"]
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(32*CH_FOLD2))
        self.Conv2 = conv_block(ch_in=int(32*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Conv3 = conv_block(ch_in=int(64*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Conv4 = conv_block(ch_in=int(128*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Conv5 = conv_block(ch_in=int(256*CH_FOLD2),ch_out=int(512*CH_FOLD2))

        self.Up5 = up_conv(ch_in=int(512*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))

        self.Up4 = up_conv(ch_in=int(256*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))
        
        self.Up3 = up_conv(ch_in=int(128*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))
        
        self.Up2 = up_conv(ch_in=int(64*CH_FOLD2),ch_out=int(32*CH_FOLD2))
        self.Up_conv2 = conv_block(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))

        self.Conv_1x1 = nn.Conv2d(int(32*CH_FOLD2),output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self, batch: Batch) -> Tensor:

        src = batch.get("sequence")

        padd_multiple = 32
        pad_len = 0
        if src.shape[1]%padd_multiple: 
            pad_len = (padd_multiple-src.shape[1]%padd_multiple)
            src = torch.cat( (src, torch.zeros((src.shape[0], pad_len), device=self.device, dtype=torch.long) ), dim=-1)

        # def get_cut_len(data_len,set_len):
        #     l = data_len
        #     if l <= set_len:
        #         l = set_len
        #     else:
        #         l = (((l - 1) // 16) + 1) * 16
        #     return l

        # pad_len = get_cut_len(src.shape[1], 80)-src.shape[1]
        # src = torch.cat( (src, torch.zeros((src.shape[0], pad_len), device=self.device, dtype=torch.long) ), dim=-1)

        x = self.seq2map(src)

        self.train()

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)

        structure = torch.transpose(d1, -1, -2) * d1

        return { 
            "structure": structure[:, :src.shape[1]-pad_len, :src.shape[1]-pad_len]
            }


    def seq2map(self, seq_int):

        # take integer encoded sequence and return last channel of embedding (pairing energy)
        def creatmat(data, device=None):

            with torch.no_grad():
                data = ''.join([int2seq[d] for d in data.tolist()])
                paired = defaultdict(float, {'AU':2., 'UA':2., 'GC':3., 'CG':3., 'UG':0.8, 'GU':0.8})

                mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
                n = len(data)

                i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing='ij')
                t = torch.arange(30).to(device)
                m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n), mat[torch.clamp(i[:,:,None]-t, 0, n-1), torch.clamp(j[:,:,None]+t, 0, n-1)], 0)
                m1 *= torch.exp(-0.5*t*t)

                m1_0pad = torch.nn.functional.pad(m1, (0, 1))
                first0 = torch.argmax((m1_0pad==0).to(int), dim=2)
                to0indices = t[None,None,:]>first0[:,:,None]
                m1[to0indices] = 0
                m1 = m1.sum(dim=2)

                t = torch.arange(1, 30).to(device)
                m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0), mat[torch.clamp(i[:,:,None]+t, 0, n-1), torch.clamp(j[:,:,None]-t, 0, n-1)], 0)
                m2 *= torch.exp(-0.5*t*t)

                m2_0pad = torch.nn.functional.pad(m2, (0, 1))
                first0 = torch.argmax((m2_0pad==0).to(int), dim=2)
                to0indices = torch.arange(29).to(device)[None,None,:]>first0[:,:,None]
                m2[to0indices] = 0
                m2 = m2.sum(dim=2)
                m2[m1==0] = 0

                return (m1+m2).to(self.device)

        # Assemble all data
        full_map = []
        one_hot_embed = torch.zeros((5, 4), device=self.device)
        one_hot_embed[1:] = torch.eye(4)
        for seq in seq_int:

            seq_hot = one_hot_embed[seq].type(torch.long)
            pair_map = torch.kron(seq_hot, seq_hot).reshape(len(seq), len(seq), 16)

            energy_map = creatmat(seq)

            full_map.append(torch.cat((pair_map, energy_map.unsqueeze(-1)), dim=-1))


        return torch.stack(full_map).permute(0, 3, 1, 2).contiguous()



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x





# class U_Net_FP(nn.Module):
#     def __init__(self,img_ch=17,output_ch=1):
#         super(U_Net_FP, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         FPNch = [8, 16, 32, 64, 128]
#         self.fpn = FP(output_ch=FPNch)

#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(32), size=3)
#         self.Conv2 = conv_block(ch_in=int(32)+FPNch[0],ch_out=int(64*CH_FOLD), size=3)
#         self.Conv3 = conv_block(ch_in=int(64*CH_FOLD)+FPNch[1],ch_out=int(128*CH_FOLD))
#         self.Conv4 = conv_block(ch_in=int(128*CH_FOLD)+FPNch[2],ch_out=int(256*CH_FOLD))
#         self.Conv5 = conv_block(ch_in=int(256*CH_FOLD)+FPNch[3],ch_out=int(512*CH_FOLD))

#         self.Up5 = tp_conv(ch_in=int(512*CH_FOLD)+FPNch[4],ch_out=int(256*CH_FOLD))
#         self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD)+FPNch[3], ch_out=int(256*CH_FOLD))

#         self.Up4 = tp_conv(ch_in=int(256*CH_FOLD),ch_out=int(128*CH_FOLD))
#         self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD)+FPNch[2], ch_out=int(128*CH_FOLD))

#         self.Up3 = tp_conv(ch_in=int(128*CH_FOLD),ch_out=int(64*CH_FOLD))
#         self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD)+FPNch[1], ch_out=int(64*CH_FOLD))

#         self.Up2 = tp_conv(ch_in=int(64*CH_FOLD),ch_out=int(32))
#         self.Up_conv2 = conv_block(ch_in=int(64)+FPNch[0], ch_out=int(32))

#         self.Conv_1x1 = nn.Conv2d(int(32),output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x, m):
#         # encoding path
#         fp1, fp2, fp3, fp4, fp5 = self.fpn(m)

#         x1 = self.Conv1(x)
#         x1 = torch.cat((x1, fp1), dim=1)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
#         x2 = torch.cat((x2, fp2), dim=1)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
#         x3 = torch.cat((x3, fp3), dim=1)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
#         x4 = torch.cat((x4, fp4), dim=1)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)
#         x5 = torch.cat((x5, fp5), dim=1)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)

#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)
#         d1 = d1.squeeze(1)
#         return torch.transpose(d1, -1, -2) * d1