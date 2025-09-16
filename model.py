from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import kl_divergence

from flow import Flow
from relational_path_gnn import RelationalPathGNN


class EmbeddingLearner(nn.Module):
    def __init__(self, emb_dim, z_dim, out_size):
        super(EmbeddingLearner, self).__init__()
        self.head_encoder = nn.Linear(emb_dim, emb_dim)
        self.tail_encoder = nn.Linear(emb_dim, emb_dim)
        self.dr = nn.Linear(z_dim, 1)

    def forward(self, h, t, r, pos_num, z):  # revise
        # z = torch.nn.functional.normalize(z, dim=-1)
        d_r = self.dr(z)
        z = z.unsqueeze(2)
        h = h + self.head_encoder(z)
        t = t + self.tail_encoder(z)
        tmp_score = torch.norm(h + r - t, 2, -1)
        score = - torch.norm(tmp_score - d_r ** 2, 2, -1)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score





class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=200, num_heads=4, chunk_size=64):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size  # 分块处理大小

        # 共享权重矩阵
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        residual = x
        B, T, H = x.size()
        
        # 单次投影获取Q/K/V
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # [B, T, h, d]
        
        # 分块处理避免大矩阵
        output = torch.zeros_like(x)
        attn_weights = torch.zeros(B, self.num_heads, T, T, device=x.device)
        
        for i in range(0, T, self.chunk_size):
            chunk_end = min(i + self.chunk_size, T)
            
            # 当前块的Q
            q_chunk = q[:, i:chunk_end].transpose(1, 2)  # [B, h, chunk_size, d]
            
            # 注意力分数 (分块计算)
            attn_scores = torch.einsum('bhid,bhjd->bhij', 
                                      q_chunk, 
                                      k.transpose(1, 2)) / math.sqrt(self.head_dim)
            
            attn_weights_chunk = F.softmax(attn_scores, dim=-1)
            
            # 更新注意力权重存储
            attn_weights[:, :, i:chunk_end, :] = attn_weights_chunk.detach()
            
            # 上下文聚合
            context = torch.einsum('bhij,bhjd->bhid', 
                                 attn_weights_chunk, 
                                 v.transpose(1, 2))
            
            # 重组输出
            output_chunk = context.transpose(1, 2).reshape(B, -1, H)
            output[:, i:chunk_end] = output_chunk
        
        # 残差连接
        output = self.norm(residual + self.out_proj(output))
        return output, attn_weights

class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size * 2,
            hidden_size=n_hidden,
            num_layers=layers,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0
        )
        
        # 高效多头注意力
        self.attention = EfficientMultiHeadAttention(embed_dim=n_hidden*2)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Linear(n_hidden*2, out_size),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        # 保持原始输入处理逻辑
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1, self.embed_size*2)
        
        # LSTM处理
        lstm_out, _ = self.lstm(inputs.permute(1, 0, 2))  # [seq_len, batch, features]
        lstm_out = lstm_out.permute(1, 0, 2)  # [batch, seq_len, features]
        
        # 应用注意力
        attn_out, _ = self.attention(lstm_out)
        
        # 特征聚合
        aggregated = 0.6 * attn_out[:, -1, :] + 0.4 * attn_out.mean(dim=1)
        
        # 最终输出
        output = self.out(aggregated)
        return output.view(batch_size, 1, 1, self.out_size)




class NPFKGC(nn.Module):
    def __init__(self, g, dataset, parameter, num_symbols, embed=None):
        super(NPFKGC, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.num_rel = len(self.rel2id)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.num_hidden1 = 500
        self.num_hidden2 = 200
        self.lstm_dim = parameter['lstm_hiddendim']
        self.lstm_layer = parameter['lstm_layers']
        self.np_flow = parameter['flow']

        self.r_path_gnn = RelationalPathGNN(g, dataset['ent2id'], len(dataset['rel2emb']), parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.r_dim = self.z_dim = 50
            self.relation_learner = LSTM_attn(embed_size=50, n_hidden=100, out_size=50, layers=2, dropout=0.5)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=50, num_hidden1=250,
                                                num_hidden2=100, r_dim=self.z_dim, dropout_p=self.dropout_p)
            self.embedding_learner = EmbeddingLearner(50, self.z_dim, 50)

        elif parameter['dataset'] == 'NELL-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=self.lstm_dim, out_size=100,
                                              layers=self.lstm_layer, dropout=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=100, num_hidden1=500,
                                                num_hidden2=200, r_dim=self.z_dim, dropout_p=self.dropout_p)
            self.embedding_learner = EmbeddingLearner(100, self.z_dim, 100)
        elif parameter['dataset'] == 'FB15K-One':
            self.r_dim = self.z_dim = 100
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=self.lstm_dim, out_size=100,
                                              layers=self.lstm_layer, dropout=self.dropout_p)
            self.latent_encoder = LatentEncoder(parameter['few'], embed_size=100, num_hidden1=500,
                                                num_hidden2=200, r_dim=self.z_dim, dropout_p=self.dropout_p)
            self.embedding_learner = EmbeddingLearner(100, self.z_dim, 100)
        if self.np_flow != 'none':
            self.flows = Flow(self.z_dim, parameter['flow'], parameter['K'])      
        self.hmc = HMCModule(self.z_dim, step_size=0.5, num_steps=10)  # 添加 HMC 模块
        self.xy_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def eval_reset(self):
        self.eval_query = None
        self.eval_z = None
        self.eval_rel = None
        self.is_reset = True

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def eval_support(self, support, support_negative, query):
        support, support_negative, query = self.r_path_gnn(support), self.r_path_gnn(support_negative), self.r_path_gnn(query)
        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)
        support_pos_r = self.latent_encoder(support, 1)
        support_neg_r = self.latent_encoder(support_negative, 0)
        target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
        target_dist = self.xy_to_mu_sigma(target_r)
        z = target_dist.sample()

        # 确保 z 需要梯度
        z.requires_grad_(True)

        z, _ = self.hmc(z, target_dist)
        if self.np_flow != 'none':
            z, _ = self.flows(z, target_dist)
        rel = self.relation_learner(support_few)
        return query, z, rel

    def eval_forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        support, support_negative, query, negative = task
        negative = self.r_path_gnn(negative)
        if self.is_reset:
            query, z, rel = self.eval_support(support, support_negative, query)
            self.eval_query = query
            self.eval_z = z
            self.eval_rel = rel
            self.is_reset = False
        else:
            query = self.eval_query
            z = self.eval_z
            rel = self.eval_rel

        # 确保 z 需要梯度
        z.requires_grad_(True)

        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)
        return p_score, n_score

    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        support, support_negative, query, negative = [self.r_path_gnn(t) for t in task]
        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative

        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)

        if iseval or istest:
            support_pos_r = self.latent_encoder(support, 1)
            support_neg_r = self.latent_encoder(support_negative, 0)
            target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            target_dist = self.xy_to_mu_sigma(target_r)
            z = target_dist.sample()
            z.requires_grad_(True)
            if self.np_flow != 'none':
                z, _ = self.flows(z, target_dist)
        else:
            query_pos_r = self.latent_encoder(query, 1)
            query_neg_r = self.latent_encoder(negative, 0)
            support_pos_r = self.latent_encoder(support, 1)
            support_neg_r = self.latent_encoder(support_negative, 0)
            context_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            target_r = torch.cat([support_pos_r, support_neg_r, query_pos_r, query_neg_r], dim=1)
            context_dist = self.xy_to_mu_sigma(context_r)
            target_dist = self.xy_to_mu_sigma(target_r)
            z = target_dist.rsample()
            z.requires_grad_(True)

            z, kld = self.hmc(z, target_dist, context_dist)
            
            if self.np_flow != 'none':
                z.requires_grad_(True)
                z, kld = self.flows(z, target_dist, context_dist)
            else:
                kld = kl_divergence(target_dist, context_dist).sum(-1)

        rel = self.relation_learner(support_few)
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)

        if iseval:
            return p_score, n_score
        else:
            return p_score, n_score, kld

class HMCModule(nn.Module):
    def __init__(self, latent_size, step_size, num_steps):
        super(HMCModule, self).__init__()
        self.latent_size = latent_size
        self.step_size = step_size
        self.num_steps = num_steps

    def forward(self, z, base_dist, prior=None):
        # Ensure z requires gradients
        z.requires_grad_(True)
        
        # Initialize momentum for dynamics simulation
        p = torch.randn_like(z)

        # Save initial state
        z_init = z.clone()
        p_init = p.clone()

        # Calculate initial Hamiltonian
        potential_energy = -base_dist.log_prob(z).sum(dim=-1)
        kinetic_energy = (p ** 2).sum(dim=-1) / 2
        hamiltonian_init = potential_energy + kinetic_energy

        # Leapfrog integration
        z, p = self.leapfrog(z, p, base_dist)

        # Calculate final Hamiltonian
        potential_energy = -base_dist.log_prob(z).sum(dim=-1)
        kinetic_energy = (p ** 2).sum(dim=-1) / 2
        hamiltonian_final = potential_energy + kinetic_energy

        # Metropolis-Hastings acceptance step
        accept_prob = torch.exp(hamiltonian_init - hamiltonian_final)
        accept = (torch.rand_like(accept_prob) < accept_prob)

        # Only accept new z and p if accept is True
        z = torch.where(accept.unsqueeze(-1), z, z_init)

        # KLD computation (if prior distribution is provided)
        kld = None
        if prior is not None:
            log_prob_prior = prior.log_prob(z).sum(dim=-1)
            log_prob_posterior = base_dist.log_prob(z).sum(dim=-1)
            kld = log_prob_posterior - log_prob_prior

        return z, kld

    def leapfrog(self, z, p, base_dist):
        """ Performs leapfrog integration for HMC. """
        for _ in range(self.num_steps):
            # 第一次动量更新
            log_prob = base_dist.log_prob(z).sum()
            grad_z = torch.autograd.grad(log_prob, z, create_graph=True)[0]
            p -= 0.5 * self.step_size * grad_z
        
            # 位置更新
            z = z + self.step_size * p
            z.requires_grad_(True)  # 重新启用梯度
        
            # 第二次动量更新
            log_prob = base_dist.log_prob(z).sum()
            grad_z = torch.autograd.grad(log_prob, z, create_graph=True)[0]
            p -= 0.5 * self.step_size * grad_z
        # Negate momentum at end of trajectory to make the proposal symmetric
        p = -p

        return z, p
        


class LatentEncoder(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, r_dim=100, dropout_p=0.5):
        super(LatentEncoder, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * embed_size + 1, num_hidden1)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, r_dim)),
            # ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs, y):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)  # (B, few, dim * 2)
        if y == 1:
            label = torch.ones(size[0], size[1], 1).to(inputs)
        else:
            label = torch.zeros(size[0], size[1], 1).to(inputs)
        x = torch.cat([x, label], dim=-1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)

        return x  # (B, few, r_dim)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def aggregate(self, r):
        return torch.mean(r, dim=1)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, few, r_dim)
        """
        r = self.aggregate(r)
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)
