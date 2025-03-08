import pdb
import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from .layers import CrossAttention
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal



def kld_gauss(mean_1, logsigma_1, mean_2, logsigma_2):
        """计算KL散度"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        q_target = Normal(mean_1, sigma_1)
        q_context = Normal(mean_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl


def reparameters(mean, logstd, mode):
    """实现重参数化技巧，使梯度能够通过随机采样过程传播"""
    sigma =  torch.exp(0.5 * logstd)
    gaussian_noise = torch.randn(mean.shape).cuda(mean.device) # 生成随机噪声，形状与均值相同
    # sampled_z = gaussian_noise * sigma + mean
    if mode == 'train':
       sampled_z = gaussian_noise * sigma + mean # 训练时应用重参数化公式
    else:
        sampled_z = mean # 直接使用均值作为采样结果
    kdl_loss = -0.5 * torch.mean(1 + logstd - mean.pow(2) - logstd.exp()) # 计算学习到的分布与标准正态分布的KL散度
    return sampled_z, kdl_loss


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # self.if_fine_tune = if_fine_tune
        # self.device = device

    def forward(self, x, att_size=7):
        # 输入的x是一个batch的图片[batch, 3, 224, 224]
        x = self.resnet.conv1(x) # 7*7卷积，输出[batch, 64, 112, 112]
        x = self.resnet.bn1(x) # 批量归一化
        x = self.resnet.relu(x) # relu激活
        x = self.resnet.maxpool(x) # 3*3最大池化层，输出[batch, 64, 56, 56]

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])

        x = self.resnet.avgpool(x) # 7*7平均池化，输出[batch, 2048, 1, 1]
        x = x.view(x.size(0), -1) # 展平的特征向量，输出[batch, 2048]

        # if not self.if_fine_tune:
        # 切断梯度反向传播路径，避免后续计算梯度回流到卷积层
        x= Variable(x.data)
        fc = Variable(fc.data)
        att = Variable(att.data)

        return x, fc, att

class HVPNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HVPNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        # 扩展BERT模型的词表大小，新增用于标记头尾实体的token(<s>、</s>、<o>、</o>)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.vis_encoding = ImageModel() 
        self.hidden_size = args.hidden_size

        self.dropout = nn.Dropout(0.5)
        # 关系分类器
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        # 获取特殊标记<s><o>对应的token ID，用于匹配
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        # 维度转换，将图像特征映射到文本特征空间，实现输出维度对齐
        self.linear = nn.Linear(2048, self.hidden_size)

        # 生成文本和图像的分布参数（包括均值和对数标准差）
        self.txt_encoding_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout) # 计算分布的中心位置
        self.txt_encoding_logstd = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout) # 计算分布的对数标准差
        self.img_encoding_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_encoding_logstd = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
        
        # 对比学习判别器
        self.score_func = self.args.score_func # 评分函数
        if self.score_func == 'bilinear':
            # 使用双线性函数作为评分函数
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        elif self.score_func == 'concat':
            # 使用拼接函数作为评分函数,简单将两个向量连接起来
            self.discrimonator = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 跨模态融合组件
        if args.fusion == 'cross':
            # 交叉注意力融合
            self.img2txt_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout) # 图像到文本
            self.txt2img_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout) # 文本到图像
        elif args.fusion == 'concat':
            # 拼接后进行线性变换
            self.cross_encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif args.fusion == 'add':
            # 相加后进行LayerNorm归一化
            self.ln = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
        mode='train'
    ):

        # 使用BERT文本编码
        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True)

        # sequence_output：每个token的上下文化表示[batch_size, seq_len, hidden_size]
        # pooler_output：基于[CLS]标记的整个序列的单一向量表示，用于句子级任务
        sequence_output, pooler_output = output.last_hidden_state, output.pooler_output
        # 获取批次大小，序列长度，隐藏层维度
        batch_size, seq_len, hidden_size = sequence_output.shape

        # 合并主图像与辅助图像
        all_images_ = torch.cat([images.unsqueeze(1), aux_imgs], dim=1) # [batch, m+1, 3, 224, 224]
        # 表示特征、全局特征、注意力特征，all_images_rep_是所有图像的ResNet特征[batch*(m+1), 2048]
        all_images_rep_, _, att_all_images = self.vis_encoding(all_images_.reshape(-1, 3, 224, 224))
        # 重塑特征形状，[batch, m+1, 2048]
        all_images = all_images_rep_.reshape(-1, self.args.m + 1, 2048)
        # 将图像特征映射到文本特征空间，实现输出维度对齐[batch, m+1, 2048] -> [batch, m+1, hidden_size]
        all_images = self.linear(all_images)

        # 变分分布参数
        txt_mean = self.txt_encoding_mean(sequence_output, sequence_output, sequence_output, attention_mask.unsqueeze(1).unsqueeze(-1)) # 自注意力计算文本分布的均值参数 [batch_size, seq_len, hidden_size]
        txt_logstd = self.txt_encoding_logstd(sequence_output, sequence_output, sequence_output, attention_mask.unsqueeze(1).unsqueeze(-1)) # 计算文本分布的对数标准差参数 [batch_size, seq_len, hidden_size]
        img_mean = self.img_encoding_mean(all_images, all_images, all_images, None) # 图像分布的均值参数 [batch_size, m+1, hidden_size]
        img_logstd = self.img_encoding_logstd(all_images, all_images, all_images, None) # 图像分布的对数标准差参数 [batch_size, m+1, hidden_size]

        # 重参数化技巧，使梯度能够通过随机采样过程传播
        sample_z_txt, txt_kdl = reparameters(txt_mean, txt_logstd, mode) # 从文本分布中采样的特征[batch, seq_len, dim]，文本分布与标准正态分布的KL散度损失
        sample_z_img, img_kdl = reparameters(img_mean, img_logstd, mode) # 从图像分布中采样的特征[batch, seq_len, dim]，图像分布与标准正态分布的KL散度损失

        # 特征聚合，将序列特征压缩成单个向量，输出形状[batch, hidden_size]
        if self.args.reduction == 'mean':
            # 对所有token/图像位置的特征进行平均池化
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.mean(dim=1), sample_z_img.mean(dim=1)
        elif self.args.reduction == 'sum':
            # 将所有token/图像位置的特征进行求和
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.sum(dim=1), sample_z_img.sum(dim=1)
        else:
            # 使用第一个token/图像位置的特征
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt[:,  0, :], sample_z_img[:, 0, :]

        # 正样本对相似度计算
        if self.score_func == 'bilinear':
            # 使用双线性函数作为评分函数
            pos_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_cls.unsqueeze(1))).squeeze(1)
        elif self.score_func == 'concat':
            # 使用拼接函数作为评分函数,简单将两个向量连接起来
            pos_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, sample_z_img_cls], dim=-1)   )  )
        # 计算正样本对的对比学习的二元交叉熵损失，使得正样本对的相似度尽可能接近1
        pos_dis_loss = nn.functional.binary_cross_entropy(pos_img_txt_score, torch.ones(pos_img_txt_score.shape).to(pos_img_txt_score.device))     
        
        # 负样本构建与损失计算
        neg_dis_loss = 0
        for s in range(1, self.args.neg_num + 1):
            # 将图像特征向量循环右移s位，构建负样本对
            neg_sample_z_img_cls = sample_z_img_cls.roll(shifts=s, dims=0)
            if self.score_func == 'bilinear':
                neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            elif self.score_func == 'concat':
                neg_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, neg_sample_z_img_cls], dim=-1)   )  )
            
            # 负样本损失计算
            neg_dis_loss_ = nn.functional.binary_cross_entropy(neg_img_txt_score, torch.zeros(neg_img_txt_score.shape).to(neg_img_txt_score.device))
            neg_dis_loss += neg_dis_loss_
        # 总的对比损失
        dis_loss = pos_dis_loss + neg_dis_loss

        # 双向跨模态融合
        out = self.img2txt_cross(sample_z_img, sample_z_txt, sample_z_txt, None) # 图像关注文本
        final_txt = self.txt2img_cross(sample_z_txt, out, out, attention_mask.unsqueeze(1).unsqueeze(-1)) # 原始文本特征与图像增强的上下文交互
        # pdb.set_trace()
        # 创建实体表示存储张量,存储两个拼接的实体向量(头实体+尾实体)
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size) # batch, 2*hidden
        for i in range(batch_size):
            # 寻找头实体位置索引
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            # 寻找尾实体位置索引
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            # 提取头实体和尾实体的隐藏状态
            head_hidden = final_txt[i, head_idx, :].squeeze()
            tail_hidden = final_txt[i, tail_idx, :].squeeze()
            # 拼接头尾实体的隐藏状态
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # 根据提取的实体向量输出关系分类预测结果，并计算分数
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(logits, labels.view(-1), reduction='sum')
            return loss, dis_loss, txt_kdl, img_kdl, logits
        return logits

class HVPNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HVPNeTNERModel, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config
        self.vis_encoding = ImageModel() 
        self.num_labels  = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(args.dropout)


        self.linear = nn.Linear(2048, args.hidden_size)
        self.txt_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.txt_logstd = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_logstd = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
       
        self.score_func = self.args.score_func
        if self.score_func == 'bilinear':
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        elif self.score_func == 'concat':
            self.discrimonator = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if args.fusion == 'cross':
            self.img2txt_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
            self.txt2img_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
        elif args.fusion == 'concat':
            self.cross_encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif args.fusion == 'add':
            self.ln = nn.LayerNorm(self.hidden_size)
       

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None, mode='train'):
        
        all_images_ = torch.cat([images.unsqueeze(1), aux_imgs], dim=1) # [batch, m+1, 3, 224, 224]
        all_images_rep_, _, att_all_images = self.vis_encoding(all_images_.reshape(-1, 3, 224, 224))
        all_images = all_images_rep_.reshape(-1, self.args.m + 1, 2048) # [batch, m+1, 2048]
        all_images = self.linear(all_images)

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(bert_output.last_hidden_state)  # bsz, len, hidden 

        txt_mean = self.txt_mean(sequence_output, sequence_output, sequence_output,attention_mask.unsqueeze(1).unsqueeze(-1))
        img_mean = self.img_mean(all_images, all_images, all_images, None)
        txt_logstd = self.txt_logstd(sequence_output, sequence_output, sequence_output,attention_mask.unsqueeze(1).unsqueeze(-1))
        img_logstd = self.img_logstd(all_images, all_images, all_images, None)

        sample_z_txt, txt_kdl = reparameters(txt_mean, txt_logstd, mode) # [batch, seq_len, dim]
        sample_z_img, img_kdl = reparameters(img_mean, img_logstd, mode) # [batch, seq_len, dim]
        
        if self.args.reduction == 'mean':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.mean(dim=1), sample_z_img.mean(dim=1)
        elif self.args.reduction == 'sum':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.sum(dim=1), sample_z_img.sum(dim=1)
        else:
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt[:,  0, :], sample_z_img[:, 0, :]
        if self.score_func == 'bilinear':
            pos_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_cls.unsqueeze(1))).squeeze(1)
        elif self.score_func == 'concat':
            pos_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, sample_z_img_cls], dim=-1)   )  )
        pos_dis_loss = nn.functional.binary_cross_entropy(pos_img_txt_score, torch.ones(pos_img_txt_score.shape).to(pos_img_txt_score.device))     
        
        neg_dis_loss = 0
        for s in range(1, self.args.neg_num + 1):
            neg_sample_z_img_cls = sample_z_img_cls.roll(shifts=s, dims=0)
            if self.score_func == 'bilinear':
                neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            elif self.score_func == 'concat':
                neg_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, neg_sample_z_img_cls], dim=-1)   )  )
            # neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            neg_dis_loss_ = nn.functional.binary_cross_entropy(neg_img_txt_score, torch.zeros(neg_img_txt_score.shape).to(neg_img_txt_score.device))
            neg_dis_loss += neg_dis_loss_
        dis_loss = pos_dis_loss + neg_dis_loss

        # sample_z_img, sample_z_txt = self.dropout(sample_z_img), self.dropout(sample_z_txt)
        out = self.img2txt_cross(sample_z_img, sample_z_txt, sample_z_txt, None)
        final_txt = self.txt2img_cross(sample_z_txt, out, out, attention_mask.unsqueeze(1).unsqueeze(-1))
        # pdb.set_trace()
        emissions = self.fc(final_txt)    # bsz, len, labels
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
            other_loss = dis_loss + txt_kdl + img_kdl
        return TokenClassifierOutput(loss=loss,logits=logits), dis_loss, txt_kdl, img_kdl
