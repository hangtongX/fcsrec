import math
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from torch import nn
from model.base.sequentialbase import SequentialRec
from model.utils.layers import attention, TimeTrsEncoder, FFN
from extra_package.zuko.flows import MAF
import torch

from model.utils.model_utils import get_mask, PreNorm


class TimeAwareMHAttn(nn.Module):
    def __init__(self, n_head, d_model, dropout_ratio):
        super(TimeAwareMHAttn, self).__init__()
        self.d_k = d_model // n_head  # 每个头的维度
        self.h = n_head  # 多头的数量

        # 三个线性层，用于输入的 query、key 和 value
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )

        # 用于时间的查询（time_k）和时间的值（time_v）的线性变换层
        self.linear_layers_time = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(2)]
        )

        # 合并所有头的输出，得到最终的输出
        self.merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)  # dropout

    def forward(self, query, key, value, time_k, time_v, mask=None):
        b = query.size(0)  # 批大小
        seq_l = time_k.size(1)  # 序列长度

        # 使用线性层对 query, key, value 进行映射
        q, k, v = [
            l(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 使用线性层对时间相关的 query (time_k), value (time_v) 进行映射
        t_k, t_v = [
            l(x)
            .view(b, seq_l, self.h, self.d_k)
            .transpose(1, 2)  # 现在去掉了时间维度中的第二个序列长度维度
            for l, x in zip(self.linear_layers_time, (time_k, time_v))
        ]

        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) + torch.matmul(
            t_k, q.transpose(-2, -1)
        )

        # 缩放并应用mask
        attention_scores = attention_scores / math.sqrt(query.size(-1)) + mask

        # 计算注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, v) + self.dropout(
            torch.matmul(attention_probs, t_v).squeeze(-2)
        )

        # 合并所有头的输出
        x = context_layer.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)

        # 最终的线性映射
        x = self.merge(x)
        return x


class TimeTrsLayer(nn.Module):
    def __init__(self, dim, n_head, exp_factor, dropout_ratio):
        super().__init__()
        self.attn_layer = TimeAwareMHAttn(n_head, dim, dropout_ratio)
        self.ffn_layer = FFN(dim, exp_factor, dropout_ratio)
        self.sublayer_1 = PreNorm(dim)
        self.sublayer_2 = PreNorm(dim)

    def forward(self, x, t, mask):
        x = self.sublayer_1(x, lambda e: self.attn_layer(e, e, e, t, t, mask))
        x = self.sublayer_2(x, self.ffn_layer)
        return x


class MixedModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(MixedModel, self).__init__()

        # 定义线性变换
        self.linear = nn.Linear(input_dim, output_dim)

        # 定义 dropout
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, A, B):
        # 假设 A 和 B 的形状为 (batch_size, n, d)

        # 在 A 上应用 Dropout
        A_dropped = self.dropout(A)  # 在训练时随机丢弃部分 A 的元素

        # 使用线性层混合 A 和 B，假设我们将其拼接后传入
        # 你可以选择其他方式混合它们，比如加法、逐元素相乘等
        mixed_input = torch.cat(
            [A_dropped, B], dim=-1
        )  # 拼接 A 和 B，形状为 (batch_size, n, 2d)

        # 通过线性层进行混合
        output = self.linear(mixed_input)  # (batch_size, n, output_dim)

        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout_ratio):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        lookup_table = self.pos_embedding.weight[: x.size(1), :]
        x = x + lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.dropout(x)
        return x


class TimeEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(TimeEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.month = nn.Embedding(13, self.latent_dim, padding_idx=0)
        self.day = nn.Embedding(32, self.latent_dim, padding_idx=0)
        self.hour = nn.Embedding(25, self.latent_dim, padding_idx=24)
        self.ffn = nn.Sequential(
            nn.Linear(self.latent_dim * 3, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )

    def forward(self, time):
        # extract time information
        m = self.month(time[:, :, 0])
        d = self.day(time[:, :, 1])
        h = self.hour(time[:, :, 2])
        time = torch.cat([m, d, h], dim=-1)
        time = self.ffn(time)
        return time


class FCSREC(SequentialRec):
    def __init__(self):
        self.trs_layer = None
        self.position = None
        self.trs_encoder = None
        self.name = self.__class__.__name__
        self.item = None
        self.latent_dim = None
        self.dropout = None
        self.time_encoder = None
        self.c_num = None
        self.backbone = None
        self.backbone_name = None
        self.mixer = None
        self.causal_flow = None
        self.hid_dims = None
        self.output_func = None
        super(FCSREC, self).__init__()

    def build(self, config=None):
        self.__dict__.update(self.model_config.to_dict())
        self.latent_dim = self.model_config.latent_dim
        self.item = nn.Embedding(
            self.model_config.item[0] + 1, self.latent_dim, padding_idx=0
        )
        self.time_encoder = TimeEncoder(latent_dim=self.latent_dim)
        self.backbone_name = self.model_config.backbone
        self.backbone = self.get_base_model(self.backbone_name)()
        self.causal_flow = MAF(
            self.c_num,
            1,
            transforms=1,
            hidden_features=self.hid_dims,
            activation=nn.ELU,
        )
        data_info = {
            "user": self.model_config.user,
            "item": self.model_config.item,
            "max_len": self.model_config.max_len,
        }
        self.position = PositionalEmbedding(
            self.model_config.max_len,
            self.latent_dim,
            self.model_config.dropout_ratio,
        )
        # self.backbone.model_config.update(data_info)
        # self.backbone.build()
        # self.mixer = MixedModel(self.latent_dim * 2, self.latent_dim)
        self.trs_layer = TimeTrsLayer(
            self.latent_dim,
            self.num_head,
            self.exp_factor,
            self.model_config.dropout_ratio,
        )
        self.trs_encoder = TimeTrsEncoder(
            self.latent_dim, self.trs_layer, self.model_config.depth
        )
        if self.model_config.out_func in self.output_funcs:
            self.output_func = nn.Linear(self.latent_dim, self.model_config.item[0] + 1)
            # self.output_func = self.output_funcs[self.model_config.out_func]
        else:
            raise IndexError(
                "Unknown output func {}".format(self.model_config.out_func)
            )

    def forward(self, data, **kwargs):
        item_with_cons, sim_loss = self.mix_with_cons_and_time(data, **kwargs)
        # mixer_output = self.backbone.calculate_scores(input=item_with_cons, data=data)
        mixer_output = self.calculate_scores(input=item_with_cons, data=data)
        if self.training:
            mixer_output.loss = mixer_output.loss + sim_loss
        return mixer_output

    def mix_with_cons_and_time(self, data, **kwargs):
        source = data["source"]
        time = data["source_time"]
        x = self.item(source)
        time = self.time_encoder(time)
        # time = time.mean(dim=2, keepdim=True).expand(-1, -1, self.latent_dim)
        # reconstruct cons based on causal flow, guidance by time and item info
        if self.training:
            cons = self.causal_flow(time.unsqueeze(-1)).rsample()
            # sim_loss = -self.causal_flow(time.unsqueeze(-1)).log_prob(cons)
            sim_loss = 0
            # print(sim_loss)
            # jac = torch.autograd.functional.jacobian(
            #     self.model.flow().transform, cons.mean(0), create_graph=True
            # )
        else:
            cons = self.causal_flow(time.unsqueeze(-1)).rsample()
            # cons = self.causal_flow().rsample()
            sim_loss = 0

        cons = cons.permute(0, 1, 3, 2)
        # reconstruct item embeddings with the influence of cons
        cons_causal_strength_on_item, _ = attention(x.unsqueeze(2), cons, cons)
        # item_with_cons = self.mixer(x, cons_causal_strength_on_item.squeeze(2))
        backs = {"item": x, "cons": cons_causal_strength_on_item.squeeze(2)}
        return backs, sim_loss

    def extra_remarks(self):
        remark = f"c_num={self.c_num}"
        sheet_name = self.backbone_name
        return {"remark": remark, "sheet_name": sheet_name}

    def calculate_scores(self, input, **kwargs):
        data = kwargs["data"]
        x = input["item"]
        time = input["cons"]
        # time = time.mean(dim=2, keepdim=True).expand(-1, -1, self.latent_dim)

        # print(time.shape)
        # raise IndexError
        x = self.position(x)
        mask = get_mask(data["source"], bidirectional=False)
        mixer_output = self.trs_encoder(x, time, mask)
        # print(mixer_output)
        # raise IndexError

        return self.output_format(mixer_output, data)

    def get_item_rep(self, input, **kwargs):
        items = input["source"]
        item_with_cons, sim_loss = self.mix_with_cons_and_time(input, **kwargs)
        x = self.position(item_with_cons["item"])
        mask = get_mask(items, bidirectional=False)
        mixer_output = self.trs_layer.attn_layer(
            x, x, x, item_with_cons["cons"], item_with_cons["cons"], mask
        )
        return mixer_output

    def plot_tsne(self):
        tsne = manifold.TSNE(
            n_components=2,
            init="pca",
            random_state=15,
            perplexity=50,
            learning_rate=500,
            n_iter=2000,
            early_exaggeration=3,
        )
        size = self.model_config.item[0] + 1
        back = None
        labels = None
        for i in range(12):
            n = 0
            while n < size:
                a = np.zeros((1, 64, 3))
                a[0, :, :] = i + 1
                time = torch.tensor(a)
                inputs = {
                    "source": torch.arange(i, min(i + 64, size)).unsqueeze(0).long(),
                    "source_time": time.long(),
                }
                label = np.full(64, i + 1)
                embs = self.get_item_rep(inputs).detach().squeeze()
                back = embs if back is None else torch.cat([back, embs])
                labels = label if labels is None else np.concatenate((labels, label))
                n = n + 64
        x_tsne = tsne.fit_transform(back.numpy())
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 36
        fig = plt.figure(figsize=(16, 16))
        scatter = plt.scatter(
            x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap="tab10", s=10, alpha=0.5
        )
        legend = plt.legend(
            *scatter.legend_elements(),
            title="Month",
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 1.20),
            frameon=True,
            edgecolor="white",
            fontsize=40,
            markerscale=6,
        )
        legend.get_frame().set_alpha(0)
        # legend.get_frame().set_linewidth(2)
        # for i in range(5):
        #     plt.scatter(
        #         x_tsne[confounders_label == i, 0],
        #         x_tsne[confounders_label == i, 1],
        #         # label=labels[i],
        #         marker="o",
        #         s=1,
        #     )
        # xtext, ytext = (
        #     np.median(x_tsne[confounders_label == i, :], axis=0)[0],
        #     np.min(x_tsne[confounders_label == i, :], axis=0)[1] - 5,
        # )
        # if i != 4:
        #     txt = plt.text(
        #         xtext, ytext, "$\epsilon_{}$".format(str(i)), fontsize=20
        #     )
        # else:
        #     txt = plt.text(xtext, ytext, "user", fontsize=20)
        # txt.set_path_effects(
        #     [
        #         Patheffects.Stroke(linewidth=5, foreground="w"),
        #         Patheffects.Normal(),
        #     ]
        # )

        # plt.legend(loc=1)
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")

        # plt.title('Visualization of confounders and user preference')
        plt.savefig("/home/hangtong/codes/plot_figures/fcsrec/tse.eps", dpi=300)

    def plot_tsne_item_with_max_circle(self):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 14
        tsne = manifold.TSNE(n_components=2, random_state=42, perplexity=10)
        reduced_embeddings = tsne.fit_transform(self.item.weight.detach().numpy())

        # 标准化数据（可选）
        scaler = StandardScaler()
        reduced_embeddings = scaler.fit_transform(reduced_embeddings)

        # 计算最小外接圆
        center = np.mean(reduced_embeddings, axis=0)  # 圆心为数据的均值
        distances = np.linalg.norm(
            reduced_embeddings - center, axis=1
        )  # 计算每个点到圆心的距离
        radius = np.max(distances)  # 半径为最大距离

        # 可视化
        plt.figure(figsize=(16, 16))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10, alpha=0.5)

        # 绘制最大圆形
        circle = plt.Circle(
            center, radius, color="r", fill=False, linestyle="--", linewidth=2
        )
        plt.gca().add_artist(circle)

        plt.xticks([])
        plt.yticks([])
        plt.axis("off")

        plt.savefig(
            "/home/hangtong/codes/plot_figures/fcsrec/tse_item_with_max_circle.eps",
            dpi=300,
        )
