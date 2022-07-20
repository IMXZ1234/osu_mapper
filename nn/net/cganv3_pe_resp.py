import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def insert_cls_prob_0(prob, insert_pos=0):
    N, L, C = prob.shape
    return torch.cat(
        [
            prob[:, :, :insert_pos],
            torch.zeros([N, L, 1], dtype=torch.float, device=prob.device),
            prob[:, :, insert_pos:],
        ], dim=-1
    )


def assemble(circle_label_prob, slider_label_prob, circle_itv=2, slider_itv=4):
    device = circle_label_prob.device
    circle_pred = torch.argmax(circle_label_prob, dim=-1).detach().cpu().numpy()
    slider_pred = torch.argmax(slider_label_prob, dim=-1).detach().cpu().numpy()
    circle_label_prob = insert_cls_prob_0(circle_label_prob, 2)
    slider_label_prob = insert_cls_prob_0(slider_label_prob, 1)
    total_len = len(circle_pred) * circle_itv
    # print('before')
    # print(pred[0])
    # print(pred)
    all_sample_revised = []
    stride = min(circle_itv, slider_itv)
    for sample_circle_pred, sample_slider_pred,\
            sample_circle_label_prob, sample_slider_label_prob in zip(
        circle_pred, slider_pred,
            circle_label_prob, slider_label_prob
    ):
        revised = []
        i = 0
        while i < total_len:
            if i % slider_itv == 0 and sample_slider_pred[i // slider_itv] == 1:
                i_slider = i // slider_itv
                while i + slider_itv < total_len and sample_slider_pred[i_slider] == 1:
                    revised.append(sample_slider_label_prob[[i_slider]])
                    pad = torch.tensor([[0, 0, 1]] * (slider_itv-1),
                                       dtype=torch.float, device=device)
                    revised.append(pad)
                    i += slider_itv
                    i_slider += 1
                # prob = label_prob[i]
                # # insert prob for label 1
                # new_prob = torch.cat(
                #     prob[0],
                #     torch.tensor([0], dtype=torch.float, device=device),
                #     prob[1],
                # )
                if i >= total_len:
                    break
                revised.append(sample_slider_label_prob[[i_slider]])
                pad = torch.tensor([[1, 0, 0]] * (stride - 1),
                                   dtype=torch.float, device=device)
                revised.append(pad)
            elif i % circle_itv == 0 and sample_circle_pred[i // circle_itv] == 1:
                i_circle = i // circle_itv
                revised.append(sample_circle_label_prob[[i_circle]])
                pad = torch.tensor([[1, 0, 0]] * (stride-1),
                                   dtype=torch.float, device=device)
                revised.append(pad)
            else:
                pad = torch.tensor([[1, 0, 0]] * stride,
                                   dtype=torch.float, device=device)
                revised.append(pad)
            i = i + stride
        all_sample_revised.append(torch.cat(revised, dim=0))
    # print('after')
    # print(torch.argmax(all_sample_revised[0], dim=-1))
    # print(all_sample_revised[0])
    return torch.stack(all_sample_revised, dim=0)


def pos_encode(length, dim, device):
    """
    ->[1(N), length, dim]
    """
    div_term = torch.tensor(
        1 / np.power(10000, (2 * (np.arange(dim)[:, np.newaxis] // 2)) / dim),
        dtype=torch.float,
        requires_grad=False,
        device=device,
    )
    position = torch.arange(
        length,
        dtype=torch.float,
        requires_grad=False,
        device=device
    ).unsqueeze(0)
    angle_rads = position * div_term
    # apply sin to even indices in the array; 2i
    angle_rads[0::2, :] = torch.sin(angle_rads[0::2, :])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[1::2, :] = torch.cos(angle_rads[1::2, :])

    pos_encoding = angle_rads.unsqueeze(0)
    return pos_encoding


def conv_block(in_feat, out_feat, kernel_size=3, normalize=True, act=True):
    layers = [nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, padding=kernel_size // 2)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    if act:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class FeatureExtractor(nn.Module):
    """
    cond_data: N, L(output_len), snap_feature(514 here) -> N, L, out_feature
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            *conv_block(in_channels, in_channels // 16, kernel_size=1, normalize=False),
            # *conv_block(in_channels // 4, in_channels // 16),
            *conv_block(in_channels // 16, out_channels, act=False),
        )

    def forward(self, cond_data):
        cond_data = cond_data.transpose(1, 2)
        cond_data = self.model(cond_data)
        return cond_data


# class FeatureExtractorShallow(nn.Module):
#     """
#     cond_data: N, L(output_len), snap_feature(514 here) -> N, L, out_feature
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         self.model = nn.Sequential(
#             *conv_block(in_channels, in_channels // 16, kernel_size=7, normalize=False),
#             # *conv_block(in_channels // 4, in_channels // 16),
#             *conv_block(in_channels // 16, out_channels, kernel_size=7, act=False),
#         )
#
#     def forward(self, cond_data):
#         cond_data = cond_data.transpose(1, 2)
#         cond_data = self.model(cond_data)
#         return cond_data


class Generator(nn.Module):
    def __init__(self, output_len, in_channels, num_classes, noise_in_channels,
                 compressed_channels=16, **kwargs):
        """
        output_snaps = output_feature_num // num_classes
        """
        super(Generator, self).__init__()

        self.circle_itv, self.slider_itv = 2, 4

        self.num_classes = num_classes
        self.output_len = output_len
        self.ext_in_channels = in_channels
        self.ext_out_channels = compressed_channels

        output_feature_num = self.output_len * self.num_classes

        # def block(in_feat, out_feat, normalize=True):
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     # if normalize:
        #     #     layers.append(nn.BatchNorm1d(out_feat, 0.8))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers
        self.ext = FeatureExtractor(self.ext_in_channels, self.ext_out_channels)
        print('G first linear in')
        print(noise_in_channels + self.ext_out_channels * self.output_len)
        print('G first linear out')
        print(16 * output_feature_num)

        self.main_in_channels = noise_in_channels + self.ext_out_channels

        self.model = nn.Sequential(
            *conv_block(self.main_in_channels, self.num_classes * 16, kernel_size=7, normalize=False),
            # *conv_block(self.num_classes * 16, self.num_classes * 16, kernel_size=7),
            # *conv_block(self.num_classes * 16, self.num_classes * 8, kernel_size=3),
            *conv_block(self.num_classes * 16, self.num_classes * 4, kernel_size=3),
            # *conv_block(self.num_classes * 4, self.num_classes * 4, kernel_size=3),
            nn.Flatten(),
            nn.Linear(self.num_classes * 4 * self.output_len, self.num_classes * 4 * self.output_len),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.circle_classifier = nn.Sequential(
            nn.Linear(self.num_classes * 4 * self.output_len, self.num_classes * self.output_len // self.circle_itv),
        )
        self.slider_classifier = nn.Sequential(
            nn.Linear(self.num_classes * 4 * self.output_len, self.num_classes * self.output_len // self.slider_itv),
        )

    def forward(self, noise, cond_data):
        # print('in G')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        # Concatenate label embedding and image to produce input
        cond_data = self.ext(cond_data)
        noise = noise.transpose(1, 2)
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        gen_input = torch.cat((cond_data, noise), dim=1)

        gen_input = pos_encode(self.output_len, self.main_in_channels, cond_data.device) + gen_input

        # N, num_classes, num_snaps -> N, num_snaps, num_classes
        # gen_output = self.model(gen_input).transpose(1, 2)
        # N * num_classes * num_snaps -> N, num_snaps, num_classes
        gen_output = self.model(gen_input).reshape([-1, self.output_len, self.num_classes])
        slider_label_prob = self.slider_classifier(gen_output)
        circle_label_prob = self.circle_classifier(gen_output)
        gen_output = assemble(circle_label_prob, slider_label_prob, self.circle_itv, self.slider_itv)
        gen_output = torch.nn.functional.gumbel_softmax(gen_output, dim=-1, hard=True)
        # N, num_snaps, num_classes
        # print('gen_output.shape')
        # print(gen_output.shape)
        return gen_output, cond_data


class Discriminator(nn.Module):
    def __init__(self, output_len, in_channels, num_classes,
                 compressed_channels=16, **kwargs):
        """
        output_feature_num = output_snaps * num_classes: generator output feature num, in one-hot format
        """
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.output_len = output_len
        self.ext_in_channels = in_channels
        self.ext_out_channels = compressed_channels

        self.ext = FeatureExtractor(self.ext_in_channels, self.ext_out_channels)

        self.main_in_channels = self.num_classes + self.ext_out_channels

        self.model = nn.Sequential(
            *conv_block(self.main_in_channels, self.num_classes * 16, kernel_size=7, normalize=False),
            *conv_block(self.num_classes * 16, self.num_classes * 8),
            *conv_block(self.num_classes * 8, self.num_classes * 2, kernel_size=3),
        )

        self.fcn = nn.Linear(self.num_classes * 2, 2)

    def forward(self, gen_output, cond_data):
        # print('in D')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('gen_output.shape')
        # print(gen_output.shape)
        cond_data = self.ext(cond_data)
        gen_output = gen_output.transpose(1, 2)
        # cond_data = self.ext(cond_data)
        # Concatenate label embedding and image to produce input
        d_in = torch.cat([gen_output, cond_data], dim=1)

        d_in = pos_encode(self.output_len, self.main_in_channels, cond_data.device) + d_in

        validity = self.model(d_in)
        validity = F.avg_pool1d(validity, validity.shape[2]).reshape([validity.shape[0], -1])

        validity = self.fcn(validity)

        # N, 2
        return validity
