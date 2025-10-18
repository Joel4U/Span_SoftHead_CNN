import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# ============================================================
# 1. SoftHeadComputer (简化版，用于测试)
# ============================================================

class SoftHeadComputer(nn.Module):
    """计算 Span 的软 Head"""
    
    def __init__(
        self,
        w_coverage=2.0,
        w_degree=1.5,
        w_pos=0.8,
        w_deprel=0.5,
        w_medoid=0.5,
        w_direction=0.3,
        temperature=0.7,
        k=3
    ):
        super().__init__()
        self.w_coverage = w_coverage
        self.w_degree = w_degree
        self.w_pos = w_pos
        self.w_deprel = w_deprel
        self.w_medoid = w_medoid
        self.w_direction = w_direction
        self.temperature = temperature
        self.k = k
    
    def compute_soft_heads(
        self,
        dep_parents: torch.Tensor,      # (bsz, L)
        token_embeds: torch.Tensor,     # (bsz, dim, L)
        pos_batch: List[List[str]],
        deprel_batch: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            soft_heads: (bsz, L, L, dim)
            head_weights: (bsz, L, L, k)
            head_indices: (bsz, L, L, k)
        """
        bsz, dim, L = token_embeds.size()
        device = token_embeds.device
        
        # 转置 token_embeds 到 (bsz, L, dim)
        token_embeds = token_embeds.transpose(1, 2)
        
        soft_heads = torch.zeros(bsz, L, L, dim, device=device)
        head_weights = torch.zeros(bsz, L, L, self.k, device=device)
        head_indices = torch.full((bsz, L, L, self.k), -1, dtype=torch.long, device=device)
        
        for b in range(bsz):
            for l in range(L):
                for r in range(l, L):
                    # 计算 span [l, r] 的头分数
                    scores = self._compute_head_scores(
                        l, r, dep_parents[b], pos_batch[b], deprel_batch[b], L
                    )
                    
                    # Softmax 归一化
                    probs = torch.softmax(scores / self.temperature, dim=0)
                    
                    # Top-K
                    topk_vals, topk_idx = torch.topk(probs, min(self.k, len(probs)))
                    
                    # 填充
                    k_actual = topk_idx.size(0)
                    head_indices[b, l, r, :k_actual] = topk_idx
                    head_weights[b, l, r, :k_actual] = topk_vals
                    
                    # 软 head 向量
                    soft_heads[b, l, r] = (
                        probs.unsqueeze(1) * token_embeds[b]
                    ).sum(dim=0)
        
        return soft_heads, head_weights, head_indices
    
    def _compute_head_scores(
        self, l: int, r: int, parents: torch.Tensor,
        pos: List[str], deprel: List[str], L: int
    ) -> torch.Tensor:
        """计算 span [l,r] 中每个 token 作为 head 的分数"""
        span_size = r - l + 1
        scores = torch.zeros(L)
        
        for t in range(l, r + 1):
            score = 0.0
            
            # 1. Coverage: 是否在 span 内
            if l <= t <= r:
                score += self.w_coverage
            
            # 2. Degree: 依存度数
            children = (parents == t).sum().item()
            score += self.w_degree * children
            
            # 3. POS 权重
            pos_weight = {
                'VERB': 2.0, 'NOUN': 1.5, 'ADJ': 1.0, 
                'ADP': 0.8, 'DET': 0.3
            }.get(pos[t], 0.5)
            score += self.w_pos * pos_weight
            
            # 4. Deprel 权重
            deprel_weight = {
                'root': 2.0, 'nsubj': 1.5, 'obj': 1.5,
                'obl': 1.2, 'amod': 1.0
            }.get(deprel[t], 0.5)
            score += self.w_deprel * deprel_weight
            
            # 5. Medoid: 到 span 中心的距离
            center = (l + r) / 2.0
            dist_to_center = abs(t - center)
            score += self.w_medoid * (1.0 / (1.0 + dist_to_center))
            
            scores[t] = score
        
        return scores


# ============================================================
# 2. 依存树距离计算
# ============================================================

def compute_dep_distance(parents: np.ndarray) -> torch.Tensor:
    """从父节点数组计算依存树距离矩阵"""
    L = len(parents)
    dist = np.full((L, L), L, dtype=np.int32)
    
    # Floyd-Warshall 算法
    # 初始化直接边
    for i in range(L):
        dist[i, i] = 0
        if parents[i] >= 0:
            p = parents[i]
            dist[i, p] = 1
            dist[p, i] = 1
    
    # 迭代更新
    for k in range(L):
        for i in range(L):
            for j in range(L):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    
    return torch.from_numpy(dist)


# ============================================================
# 3. SpanNeighborBuilder (修改版)
# ============================================================

class SpanNeighborBuilder(nn.Module):
    """基于依存图的 Span 邻居构建器"""
    
    def __init__(
        self,
        K: int = 10,
        Ktok: int = 3,
        d: int = 2,
        gamma: float = 1.0,
        max_width: Optional[int] = None,
        cap_spans_per_token: int = 128,
        cap_cands: int = 512,
        use_self_loop: bool = True,
        use_geom_fill: bool = True,
    ):
        super().__init__()
        self.K = K
        self.Ktok = Ktok
        self.d = d
        self.gamma = gamma
        self.max_width = max_width
        self.cap_spans_per_token = cap_spans_per_token
        self.cap_cands = cap_cands
        self.use_self_loop = use_self_loop
        self.use_geom_fill = use_geom_fill
    
    def forward(
        self,
        head_indices: torch.Tensor,
        head_weights: torch.Tensor,
        dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        输入:
            head_indices: (bsz, L, L, Ktok)
            head_weights: (bsz, L, L, Ktok)
            dist: (bsz, L, L)
        返回:
            N_idx: (bsz, S, K)
            N_mask: (bsz, S, K)
            span_maps: List[(span2id, id2lr)]
        """
        bsz = head_indices.size(0)
        
        all_N_idx = []
        all_N_mask = []
        span_maps = []
        
        for b in range(bsz):
            N_idx, N_mask, span2id, id2lr = self._build_for_sentence(
                b, head_indices, head_weights, dist
            )
            all_N_idx.append(N_idx)
            all_N_mask.append(N_mask)
            span_maps.append((span2id, id2lr))
        
        # Padding 到相同长度
        max_S = max(n.size(0) for n in all_N_idx)
        N_idx_padded = []
        N_mask_padded = []
        
        for n_idx, n_mask in zip(all_N_idx, all_N_mask):
            S = n_idx.size(0)
            if S < max_S:
                pad_size = max_S - S
                n_idx = torch.cat([
                    n_idx,
                    torch.full((pad_size, self.K), -1, dtype=torch.long, device=n_idx.device)
                ], dim=0)
                n_mask = torch.cat([
                    n_mask,
                    torch.zeros((pad_size, self.K), dtype=torch.bool, device=n_mask.device)
                ], dim=0)
            N_idx_padded.append(n_idx)
            N_mask_padded.append(n_mask)
        
        N_idx = torch.stack(N_idx_padded, dim=0)
        N_mask = torch.stack(N_mask_padded, dim=0)
        
        return N_idx, N_mask, span_maps
    
    def _build_for_sentence(self, b, head_indices, head_weights, dist):
        """单句处理"""
        device = dist.device
        L = head_indices.size(1)
        
        # 1. Span ID 映射
        span2id, id2lr = self._build_span_id_maps(L, device)
        S = id2lr.size(0)
        
        # 2. Token 邻接
        neighbors = self._build_token_neighbors(dist[b])
        
        # 3. Token → Span 反向索引
        top1_heads = head_indices[b, :, :, 0]
        spans_by_head = self._build_spans_by_head(top1_heads, span2id)
        
        # 4. 初始化邻居容器
        N_idx = torch.full((S, self.K), -1, dtype=torch.long, device=device)
        N_mask = torch.zeros((S, self.K), dtype=torch.bool, device=device)
        
        # 5. 为每个 span 构建邻居
        for sid in range(S):
            l1, r1 = id2lr[sid].tolist()
            
            # 获取 head tokens
            hidx = head_indices[b, l1, r1, :self.Ktok]
            hidx = hidx[hidx >= 0].tolist()
            
            # 收集候选
            candidate_sids = self._collect_candidates(
                sid, hidx, neighbors, spans_by_head
            )
            
            # 门控打分
            if len(candidate_sids) > 0:
                cand_tensor = torch.tensor(
                    candidate_sids, dtype=torch.long, device=device
                )
                gate_scores = self._score_candidates_gate(
                    b, l1, r1, cand_tensor, id2lr,
                    head_indices, head_weights, dist
                )
                
                # Top-K
                topk_num = min(self.K, gate_scores.numel())
                if topk_num > 0:
                    topk_vals, topk_idx = torch.topk(gate_scores, topk_num)
                    sel = cand_tensor[topk_idx]
                    N_idx[sid, :topk_num] = sel
                    N_mask[sid, :topk_num] = True
            
            # 补齐
            self._fill_neighbors(sid, l1, r1, L, N_idx, N_mask, span2id)
        
        return N_idx, N_mask, span2id, id2lr
    
    def _build_span_id_maps(self, L, device):
        """构建 span ↔ id 映射"""
        span2id = torch.full((L, L), -1, dtype=torch.long, device=device)
        id2lr = []
        sid = 0
        
        for l in range(L):
            rmax = L - 1 if self.max_width is None else min(L - 1, l + self.max_width - 1)
            for r in range(l, rmax + 1):
                span2id[l, r] = sid
                id2lr.append((l, r))
                sid += 1
        
        id2lr = torch.tensor(id2lr, dtype=torch.long, device=device)
        return span2id, id2lr
    
    def _build_token_neighbors(self, dist):
        """构建 token 邻接"""
        L = dist.size(0)
        neighbors = []
        
        for t in range(L):
            nbr = torch.nonzero(dist[t] <= self.d, as_tuple=False).squeeze(-1)
            nbr_list = nbr.tolist() if nbr.numel() > 0 else []
            if t not in nbr_list:
                nbr_list.append(t)
            neighbors.append(nbr_list)
        
        return neighbors
    
    def _build_spans_by_head(self, top1_heads, span2id):
        """构建 token → spans 反向索引"""
        L = top1_heads.size(0)
        spans_by_head = [[] for _ in range(L)]
        
        for l in range(L):
            for r in range(l, L):
                sid = span2id[l, r].item()
                if sid < 0:
                    continue
                
                u = int(top1_heads[l, r].item())
                if u < 0:
                    continue
                
                spans_by_head[u].append(sid)
        
        # 裁剪
        for u in range(L):
            if len(spans_by_head[u]) > self.cap_spans_per_token:
                spans_by_head[u] = spans_by_head[u][:self.cap_spans_per_token]
        
        return spans_by_head
    
    def _collect_candidates(self, sid, hidx, neighbors, spans_by_head):
        """收集候选 spans"""
        candidate_sids_set = set()
        
        if len(hidx) > 0:
            U = set()
            for t in hidx:
                U.update(neighbors[t])
            
            for u in U:
                for ss in spans_by_head[u]:
                    if ss != sid:
                        candidate_sids_set.add(ss)
        
        cand = list(candidate_sids_set)
        if len(cand) > self.cap_cands:
            cand = cand[:self.cap_cands]
        
        return cand
    
    def _score_candidates_gate(self, b, l1, r1, cand_sids, id2lr,
                               head_indices, head_weights, dist):
        """门控打分"""
        if cand_sids.numel() == 0:
            return torch.empty(0, device=dist.device)
        
        device = dist.device
        
        # s1 heads
        idx1 = head_indices[b, l1, r1]
        w1 = head_weights[b, l1, r1]
        mask1 = (idx1 >= 0)
        idx1 = idx1[mask1]
        w1 = w1[mask1]
        
        if idx1.numel() == 0:
            return torch.zeros(cand_sids.size(0), device=device)
        
        k1 = idx1.size(0)
        
        # s2 heads (批量)
        l2r2 = id2lr[cand_sids]
        l2 = l2r2[:, 0]
        r2 = l2r2[:, 1]
        
        idx2 = head_indices[b, l2, r2]
        w2 = head_weights[b, l2, r2]
        mask2 = (idx2 >= 0).float()
        
        # 距离张量
        idx2_safe = idx2.clamp(min=0)
        D_list = []
        for i in range(k1):
            t = idx1[i].item()
            D_row = dist[b, t, idx2_safe]
            D_list.append(D_row)
        D = torch.stack(D_list, dim=0)
        
        # 门控
        G = torch.exp(-self.gamma * D.float())
        G = G * mask2.unsqueeze(0)
        
        # 加权求和
        w1_exp = w1.view(-1, 1, 1)
        tmp = (w1_exp * G).sum(dim=0)
        gate = (tmp * w2 * mask2).sum(dim=-1)
        
        return gate
    
    def _fill_neighbors(self, sid, l1, r1, L, N_idx, N_mask, span2id):
        """补齐邻居"""
        device = N_idx.device
        fill_needed = self.K - N_mask[sid].sum().item()
        
        if fill_needed <= 0:
            return
        
        # 自环
        if self.use_self_loop:
            N_idx[sid, self.K - fill_needed] = sid
            N_mask[sid, self.K - fill_needed] = True
            fill_needed -= 1
        
        # 几何 8 邻域
        if fill_needed > 0 and self.use_geom_fill:
            geom = []
            for dl in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dl == 0 and dr == 0:
                        continue
                    nl, nr = l1 + dl, r1 + dr
                    if 0 <= nl <= nr < L:
                        ss = span2id[nl, nr].item()
                        if ss >= 0 and ss != sid:
                            existing = N_idx[sid][N_mask[sid]]
                            if not torch.any(existing == ss):
                                geom.append(ss)
                                if len(geom) >= fill_needed:
                                    break
                if len(geom) >= fill_needed:
                    break
            
            if len(geom) > 0:
                empty_pos = torch.nonzero(~N_mask[sid], as_tuple=False).squeeze(-1)
                fill_num = min(len(geom), empty_pos.numel())
                N_idx[sid, empty_pos[:fill_num]] = torch.tensor(
                    geom[:fill_num], dtype=torch.long, device=device
                )
                N_mask[sid, empty_pos[:fill_num]] = True


# ============================================================
# 4. 可视化函数
# ============================================================

def visualize_dependency_tree(tokens, parents, ax):
    """可视化依存树"""
    import networkx as nx
    
    G = nx.DiGraph()
    edges = []
    for i, p in enumerate(parents):
        if p >= 0:
            edges.append((p, i))
    
    G.add_edges_from(edges)
    
    # 层次布局
    root = np.where(parents == -1)[0][0]
    pos = hierarchy_pos(G, root)
    
    # 归一化位置
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        for node in pos:
            x, y = pos[node]
            pos[node] = (
                (x - x_min) / (x_max - x_min + 1e-9),
                (y - y_min) / (y_max - y_min + 1e-9)
            )
    
    # 绘制
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                           arrows=True, arrowsize=15, ax=ax)
    
    labels = {i: f"{tokens[i]}\n({i})" for i in range(len(tokens))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title("Dependency Tree", fontweight='bold')
    ax.axis('off')


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """计算层次布局位置"""
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                   pos=None, parent=None, parsed=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    if parsed is None:
        parsed = [root]
    else:
        parsed.append(root)
    
    children = list(G.neighbors(root))
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                pos=pos, parent=root, parsed=parsed)
    return pos


def visualize_span_neighbors(tokens, target_span, neighbors, id2lr, ax):
    """可视化 span 邻居关系"""
    import networkx as nx
    
    l, r = target_span
    target_text = " ".join(tokens[l:r+1])
    
    G = nx.Graph()
    target_sid = None
    
    # 找到目标 span id
    for sid, (sl, sr) in enumerate(id2lr):
        if sl == l and sr == r:
            target_sid = sid
            break
    
    if target_sid is None:
        ax.text(0.5, 0.5, "Target span not found", ha='center', va='center')
        ax.axis('off')
        return
    
    # 添加节点和边
    for sid in neighbors:
        G.add_node(sid)
        if sid != target_sid:
            G.add_edge(target_sid, sid)
    
    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 节点颜色
    node_colors = ['#FF6B6B' if sid == target_sid else '#4ECDC4' 
                   for sid in neighbors]
    
    # 绘制
    nx.draw_networkx_nodes(G, pos, nodelist=neighbors,
                           node_color=node_colors, node_size=1200,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#95A5A6',
                           width=2, alpha=0.6, ax=ax)
    
    # 标签
    labels = {}
    for sid in neighbors:
        sl, sr = id2lr[sid]
        text = " ".join(tokens[sl:sr+1])
        if len(text) > 12:
            text = text[:12] + "..."
        labels[sid] = f"[{sl},{sr}]\n{text}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    
    ax.set_title(f"Neighbors of [{l},{r}] \"{target_text}\"",
                fontweight='bold', fontsize=10)
    ax.axis('off')


def print_neighbor_details(tokens, target_span, neighbors, id2lr, 
                          gate_scores=None):
    """打印邻居详情"""
    l, r = target_span
    target_text = " ".join(tokens[l:r+1])
    
    print("\n" + "="*80)
    print(f"Target Span: [{l},{r}] \"{target_text}\"")
    print("="*80)
    
    print(f"\n{'Rank':<6}{'Span ID':<10}{'Range':<12}{'Text':<25}{'Gate Score':<12}")
    print("-"*80)
    
    for rank, sid in enumerate(neighbors, 1):
        sl, sr = id2lr[sid]
        text = " ".join(tokens[sl:sr+1])
        
        flag = ""
        if sl == l and sr == r:
            flag = "★ SELF"
        elif gate_scores and sid in gate_scores:
            score = gate_scores[sid]
            flag = f"{score:.4f}"
        else:
            flag = "GEOM"
        
        print(f"{rank:<6}{sid:<10}[{sl},{sr}]{'':8}{text:<25}{flag:<12}")


# ============================================================
# 5. 完整测试
# ============================================================

def test_complete_pipeline():
    """完整测试流程"""
    
    print("="*80)
    print("Span Neighbor Builder - Complete Test")
    print("="*80)
    
    # ===== 测试数据 =====
    tokens = ["The","quick","brown","fox","jumps","over","the","lazy","dog"]
    n = len(tokens)
    
    parents_np = np.array([3,3,3,4,-1,8,8,8,4])
    pos = ["DET","ADJ","ADJ","NOUN","VERB","ADP","DET","ADJ","NOUN"]
    deprel = ["det","amod","amod","nsubj","root","case","det","amod","obl"]
    
    print(f"\nSentence: {' '.join(tokens)}")
    print(f"Parents:  {parents_np.tolist()}")
    print(f"POS:      {pos}")
    print(f"DepRel:   {deprel}")
    
    # ===== 准备输入 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    bsz = 1
    dim = 64
    
    # 依存父节点
    dep_parents = torch.from_numpy(parents_np).unsqueeze(0).to(device)  # (1, 9)
    
    # Token embeddings (随机)
    token_embeds = torch.randn(bsz, dim, n, device=device)
    
    # 批次化的 POS 和 DepRel
    pos_batch = [pos]
    deprel_batch = [deprel]
    
    # ===== 计算依存距离 =====
    print("\n" + "-"*80)
    print("Computing dependency distance...")
    dist_matrix = compute_dep_distance(parents_np)
    dist = dist_matrix.unsqueeze(0).to(device)  # (1, 9, 9)
    
    print("\nDependency Distance Matrix:")
    print(dist_matrix.numpy())
    
    # ===== 计算 Soft Heads =====
    print("\n" + "-"*80)
    print("Computing soft heads...")
    
    computer = SoftHeadComputer(
        w_coverage=2.0,
        w_degree=1.5,
        w_pos=0.8,
        w_deprel=0.5,
        w_medoid=0.5,
        w_direction=0.3,
        temperature=0.7,
        k=3
    )
    
    soft_heads, head_weights, head_indices = computer.compute_soft_heads(
        dep_parents, token_embeds, pos_batch, deprel_batch
    )
    
    print(f"\nSoft heads shape: {soft_heads.shape}")      # (1, 9, 9, 64)
    print(f"Head weights shape: {head_weights.shape}")    # (1, 9, 9, 3)
    print(f"Head indices shape: {head_indices.shape}")    # (1, 9, 9, 3)
    
    # 示例：查看 span [4,6] "jumps over the" 的 heads
    target_l, target_r = 4, 6
    print(f"\nSpan [{target_l},{target_r}] \"{' '.join(tokens[target_l:target_r+1])}\":")
    print(f"  Top heads indices: {head_indices[0, target_l, target_r].tolist()}")
    print(f"  Top heads weights: {head_weights[0, target_l, target_r].tolist()}")
    
    # ===== 构建 Span 邻居 =====
    print("\n" + "-"*80)
    print("Building span neighbors...")
    
    builder = SpanNeighborBuilder(
        K=10,
        Ktok=3,
        d=2,
        gamma=1.0,
        max_width=None,
    )
    
    N_idx, N_mask, span_maps = builder(head_indices, head_weights, dist)
    
    print(f"\nNeighbor indices shape: {N_idx.shape}")  # (1, S, 10)
    print(f"Neighbor mask shape: {N_mask.shape}")
    
    # ===== 分析目标 Span =====
    span2id, id2lr = span_maps[0]
    
    # 找到目标 span 的 ID
    target_sid = span2id[target_l, target_r].item()
    print(f"\nTarget span ID: {target_sid}")
    
    # 获取邻居
    neighbors_mask = N_mask[0, target_sid]
    neighbors_ids = N_idx[0, target_sid][neighbors_mask].tolist()
    
    print(f"Number of neighbors: {len(neighbors_ids)}")
    
    # 打印邻居详情
    print_neighbor_details(
        tokens, (target_l, target_r), neighbors_ids,
        [(id2lr[i, 0].item(), id2lr[i, 1].item()) for i in range(id2lr.size(0))]
    )
    
    # ===== 可视化 =====
    print("\n" + "-"*80)
    print("Generating visualizations...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 依存树
    visualize_dependency_tree(tokens, parents_np, ax1)
    
    # Span 邻居图
    id2lr_list = [(id2lr[i, 0].item(), id2lr[i, 1].item()) 
                  for i in range(id2lr.size(0))]
    visualize_span_neighbors(
        tokens, (target_l, target_r), neighbors_ids, id2lr_list, ax2
    )
    
    plt.tight_layout()
    plt.savefig('span_neighbors_test.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: span_neighbors_test.png")
    
    # ===== 统计信息 =====
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    
    total_spans = id2lr.size(0)
    avg_neighbors = N_mask[0].float().sum(-1).mean().item()
    
    print(f"Total spans: {total_spans}")
    print(f"Average neighbors per span: {avg_neighbors:.2f}")
    
    # 邻居类型分布
    self_loops = 0
    for sid in range(total_spans):
        if sid in N_idx[0, sid][N_mask[0, sid]]:
            self_loops += 1
    
    print(f"Spans with self-loop: {self_loops} / {total_spans} ({100*self_loops/total_spans:.1f}%)")
    
    plt.show()
    
    return {
        'tokens': tokens,
        'target_span': (target_l, target_r),
        'neighbors': neighbors_ids,
        'N_idx': N_idx,
        'N_mask': N_mask,
        'span_maps': span_maps,
    }


if __name__ == "__main__":
    results = test_complete_pipeline()
