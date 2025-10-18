from torch import nn
import torch
import torch.nn.functional as F
from .cnn import MaskCNN
from .gsda import GlobalSDAware
# from .GE import GlobalSDAware
from .multi_head_biaffine import MultiHeadBiaffine
from .embedder import PLMEmbedder
# from .fusion import CGAFusion
from .gate_fusion import UnifiedSpanFusion
from .span_softhead import SoftHeadComputer
# from .span_neighbor_processor import SpanNeighborBuilder


class CNNNer(nn.Module):
    def __init__(self, bert_name , num_rel_tag, num_ner_tag, postag2id, deplabel2id, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, n_head=4, cnn_depth=3):
        super(CNNNer, self).__init__()
        self.embedder = PLMEmbedder(encoder_name=bert_name)
        # self.rel_embedding = nn.Embedding(num_rel_tag + 1, embedding_dim=25, padding_idx=-2) # 51个位置：50个关系 + 1个padding
        emb_dim = self.embedder.get_output_dim()

        if size_embed_dim!=0:
            n_pos = 50
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        
        self.cnn_dim = cnn_dim
        biaffine_input_size = emb_dim

        self.head_mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(biaffine_input_size, biaffine_size), 
                                      nn.GELU() # nn.LeakyReLU(),
                                      )
        self.tail_mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(biaffine_input_size, biaffine_size), 
                                      nn.GELU() # nn.LeakyReLU(),
                                      )

        self.dropout = nn.Dropout(0.4)
        if n_head>0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth>0:
            self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=3, depth=cnn_depth)

        self.down_fc = nn.Linear(cnn_dim, num_ner_tag)
        self.logit_drop = logit_drop
        self.num_ner_tag = num_ner_tag
        self.get_softheads = SoftHeadComputer(w_coverage=2.0, w_degree=1.5, w_pos=0.8, w_deprel=0.5, w_medoid=0.5, w_direction=0.3, 
                                              temperature=0.7, k=3, learnable=False  # 如果需要学习权重，设为 True
                                              )
        self.max_span_width = 32
        # Soft heads 分支
        self.use_soft_heads = True
        self.softhead_proj = nn.Linear(emb_dim, hsz, bias=False)
        self.softhead_fuse_ln = nn.LayerNorm(hsz)
        self.softhead_fuse_dropout = nn.Dropout(0.1)
        self.softhead_alpha = nn.Parameter(torch.tensor(0.0)) 
        # 可学习门控，初始化为 0 使训练初期贡献为 sigmoid(0)=0.5
        self.soft_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask, orig_to_tok_index, heads, rels, precomputed, matrix=None): 
        word_rep = self.embedder(input_ids, orig_to_tok_index, attention_mask) # (bsz, L, dim)
        # rel_emb = self.rel_embedding(rels)
        # word_rep = torch.cat((word_rep, rel_emb), dim=-1).contiguous()
        head_state = self.head_mlp(word_rep)
        tail_state = self.tail_mlp(word_rep)

        if hasattr(self, 'U'):
            biaf_scores = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            biaf_scores = self.multi_head_biaffine(head_state, tail_state)

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:word_rep.size(1), :word_rep.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(word_rep.size(0), -1, -1, -1)], dim=-1)
        # size_scores = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        # scores = size_scores + biaf_scores

        max_width_soft_heads, max_width_head_weights, max_width_head_indices = self.get_softheads(word_rep, heads, precomputed)
        soft_heads, _, _ = self.get_softheads.expand_to_full_matrix(max_width_soft_heads, max_width_head_weights, max_width_head_indices)

        if getattr(self, 'use_soft_heads', False):
            soft_feat = self.softhead_proj(soft_heads)  # (B, L, L, hsz)
            affined_cat = self.softhead_fuse_ln(affined_cat + self.softhead_alpha * self.softhead_fuse_dropout(soft_feat))  # (B, L, L, hsz)
        
        softhead_scores = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)    # (bsz, num_labels, L, L)
        scores = softhead_scores + biaf_scores
        # CNN
        if hasattr(self, 'cnn'):
            lengths = (orig_to_tok_index != 0).sum(dim=-1)
            mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1) # bsz x length x length
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(u_scores, p=self.logit_drop, training=self.training)
            # bsz, num_label, max_len, max_len = u_scores.size()
            u_scores = self.cnn(u_scores, pad_mask)
            scores = u_scores + scores
        scores = self.down_fc(scores.permute(0, 2, 3, 1))
        if self.training:
            assert scores.size(-1) == matrix.size(-1)
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
            # valid = matrix.ne(-100)  # 布尔掩码：有效标签
            # if valid.any():
            #     loss = F.binary_cross_entropy_with_logits(scores[valid], matrix[valid].float(), reduction='mean')
            # else:
            #     loss = scores.sum() * 0.0
            return loss
        return scores