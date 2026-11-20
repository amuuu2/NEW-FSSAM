"""
Parameter-free Reverse-Cycle & Hard-Negative Calibration operators for FSSAM.
"""
import torch
import torch.nn.functional as F


def min_max_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Min-max normalize tensor over its spatial dimensions (H, W)."""
    b, c, h, w = x.shape
    x_flat = x.view(b, c, -1)
    mn = x_flat.min(-1, keepdim=True)[0]
    mx = x_flat.max(-1, keepdim=True)[0]
    return ((x_flat - mn) / (mx - mn + eps)).view(b, c, h, w)


def weighted_gap(feat: torch.Tensor, mask: torch.Tensor, eps: float = 0.0005) -> torch.Tensor:
    """Masked global average pooling."""
    feat = feat * mask
    h, w = feat.shape[-2:]
    area = F.avg_pool2d(mask, (h, w)) * h * w + eps
    return F.avg_pool2d(feat, (h, w)) * h * w / area


def cos_sim_map(qry_feat: torch.Tensor, proto: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pixel-wise cosine similarity between query feature map and prototype."""
    q_norm = F.normalize(qry_feat, p=2, dim=1, eps=eps)
    p_norm = F.normalize(proto, p=2, dim=1, eps=eps)
    return (q_norm * p_norm).sum(dim=1, keepdim=True)


def compute_reverse_cycle(
    qry_feat: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
    temp: float = 25.0,
):
    """
    Compute pixel-wise Query-to-Support Reverse-Cycle Consistency C_{Q->S} in [0, 1]
    and support quality metrics (SNR margin & reverse projection precision).
    """
    b, _, h, w = qry_feat.shape
    q_norm = F.normalize(qry_feat.flatten(2), p=2, dim=1)
    s_norm = F.normalize(sup_feat.flatten(2), p=2, dim=1)
    sim_q2s = torch.bmm(q_norm.transpose(1, 2), s_norm)

    attn_q2s = F.softmax(sim_q2s * temp, dim=-1)
    m_s_flat = sup_mask.view(b, -1, 1)
    cycle_prob = torch.bmm(attn_q2s, m_s_flat).view(b, 1, h, w)
    cycle_norm = min_max_norm(cycle_prob)

    fg_proto = weighted_gap(sup_feat, sup_mask)
    bg_proto = weighted_gap(sup_feat, 1.0 - sup_mask)
    raw_fg = cos_sim_map(qry_feat, fg_proto)
    raw_bg = cos_sim_map(qry_feat, bg_proto)

    # Top-10% foreground-background cosine margin (SNR proxy)
    raw_margin = F.relu(raw_fg - raw_bg).view(b, -1)
    top_k = max(1, raw_margin.size(1) // 10)
    snr_val = raw_margin.topk(top_k, dim=-1)[0].mean(dim=-1)

    # Reverse projection precision inside support foreground mask
    fg_n = min_max_norm(raw_fg)
    bg_n = min_max_norm(raw_bg)
    disc_n = min_max_norm(F.relu(fg_n - 0.80 * (1.0 - 0.25 * cycle_norm) * bg_n))
    q_seed_proto = weighted_gap(qry_feat, (disc_n > 0.5).float() * disc_n + 1e-4)
    rev_sim_s = min_max_norm(F.relu(cos_sim_map(sup_feat, q_seed_proto) - cos_sim_map(sup_feat, bg_proto)))
    rev_prec = (rev_sim_s * sup_mask).sum(dim=(1, 2, 3)) / (rev_sim_s.sum(dim=(1, 2, 3)) + 1e-5)

    return cycle_prob, cycle_norm, snr_val, rev_prec


def compute_rc_disc_prior(
    fg_corr: torch.Tensor,
    bg_corr: torch.Tensor,
    cycle_map: torch.Tensor,
    lam: float = 0.80,
    beta: float = 0.25,
) -> torch.Tensor:
    """
    Reverse-cycle modulated discriminative prior:
    P_Q^Disc = Norm(ReLU(P_Q^FG - lam * (1 - beta * C_{Q->S}) * P_Q^BG)).
    """
    disc = F.relu(fg_corr - lam * (1.0 - beta * cycle_map) * bg_corr)
    return min_max_norm(disc)


def compute_hard_bg_prior(
    fg_corr: torch.Tensor,
    disc_corr: torch.Tensor,
    cycle_map: torch.Tensor,
) -> torch.Tensor:
    """
    Extract query Hard-BG false positive prior from high FG-similarity but low cycle-consistency regions.
    """
    return min_max_norm(F.relu(fg_corr - disc_corr) * (1.0 - cycle_map))


def bi_mem_refine_step(
    generate_prior_mem_fn,
    sup_fg_mem_feats: torch.Tensor,
    qry_fg_mem_feats: torch.Tensor,
    qry_ae_mem_feats: torch.Tensor,
    qry_ae_mem_init: torch.Tensor,
    sup_fg: torch.Tensor,
    prior_fg: torch.Tensor,
    prior_ae: torch.Tensor,
    size,
    cycle_32: torch.Tensor,
    hard_bg_32: torch.Tensor,
    prior_mem_qry_sup_fg: torch.Tensor = None,
    p_q_neg: torch.Tensor = None,
):
    """
    Single step of Bidirectional Iterative Memory Refinement (Bi-IMR):
    combines cycle-protected FG fusion with manifold-preserving soft BG erasure.
    """
    if prior_mem_qry_sup_fg is None:
        prior_mem_qry_sup_fg = generate_prior_mem_fn(qry_fg_mem_feats, sup_fg_mem_feats, sup_fg, size)
    if p_q_neg is None:
        p_q_neg = generate_prior_mem_fn(qry_fg_mem_feats, qry_fg_mem_feats, hard_bg_32 + 1e-4, size)

    p_qq = generate_prior_mem_fn(qry_fg_mem_feats, qry_ae_mem_feats, prior_ae, size)
    penalty_sup = 0.88 * (1.0 - 0.20 * cycle_32) * (1.0 - prior_mem_qry_sup_fg) + 0.12 * F.relu(p_q_neg - prior_mem_qry_sup_fg)
    p_fg = F.relu(p_qq - penalty_sup)
    g_erase = 0.06 * F.relu(p_q_neg - prior_mem_qry_sup_fg) * (1.0 - cycle_32)

    qry_ae_mem_feats = (1.0 - g_erase) * qry_ae_mem_feats + g_erase * qry_ae_mem_init
    qry_ae_mem_feats = qry_fg_mem_feats * p_fg + qry_ae_mem_feats * (1.0 - p_fg)
    prior_ae = min_max_norm(prior_fg * p_fg + (1.0 - g_erase) * prior_ae * (1.0 - p_fg))

    return qry_ae_mem_feats, prior_ae, prior_mem_qry_sup_fg, p_q_neg
