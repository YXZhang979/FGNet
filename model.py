import model.resnet as resnet
import model.FTM as FTM
import model.FEM as FEM
import model.Head as Head

import torch
from torch import nn
import torch.nn.functional as F


class FGNetPlus(nn.Module):
    def __init__(self, backbone, refine=False, num_class=21):
        super(FGNetPlus, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.refine = refine
        self.mu = 0.7
        self.tau = 0.5
        self.beta1, self.beta2 = 0.5, 0.5
        self.num_class = num_class

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):

        h, w = img_q.shape[-2:]
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])
                s_0 = self.layer1(s_0)
            s_0 = self.layer2(s_0)
            s_0 = self.layer3(s_0)
            feature_s_list.append(s_0)
            del s_0

        with torch.no_grad():
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)

        back_proto_list = get_back_protos(feature_s_list, img_s_list, mask_s_list)

        proto_list = []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            proto = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())
            proto_list.append(proto)

            if self.training:
                supp_similarity = F.cosine_similarity(feature_s_list[k], proto.unsqueeze(-1).unsqueeze(-1),
                                                      dim=1)
                supp_out = F.interpolate(supp_similarity, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        proto_s = torch.mean(torch.stack(proto_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        query_mask = F.cosine_similarity(feature_q, proto_s, dim=1)
        proto_q = self.masked_average_pooling(feature_q, (query_mask > self.mu).float())
        proto_f = self.beta1 * proto_s + self.beta2 * proto_q.unsqueeze(-1).unsqueeze(-1)

        proto_f_exp = proto_f.expand(-1, -1, h, w)
        query_mask2 = F.cosine_similarity(feature_q, proto_f, dim=1)

        ftm = FTM()
        agn_s, agn_q = ftm(feature_q)
        query_mask2 = query_mask2.unsqueeze(1)

        feat_cat = torch.cat([feature_q, proto_f_exp, query_mask2, agn_q], dim=1)
        fem = FEM(in_channel=feat_cat.shape[1])
        feat_enh = fem(feat_cat)
        query_mask3 = F.cosine_similarity(feat_enh, proto_f, dim=1)
        proto_e = self.masked_average_pooling(feat_enh, (query_mask3 > self.tau).float())
        pred_1 = F.cosine_similarity(feature_q, proto_e.unsqueeze(-1).unsqueeze(-1), dim=1)
        pred_1 = pred_1.unsqueeze(1)
        seg_head = Head(feature_q.shape[1])
        pred_2 = seg_head(feat_enh)
        pred = 0.5 * pred_1 + 0.5 * pred_2

        return pred, proto_e, proto_q, proto_s, proto_f, agn_s, agn_q, back_proto_list

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature