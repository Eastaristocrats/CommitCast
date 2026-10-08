"""Public facade for the server's original settled-risk gate."""
from layers.online_ridge import strict_online_discounted_ols_gate_mix


def settled_risk_exposure(target, checkpoint, proposal, *, segment_len,
                          warmup=16, half_life_rows=512., ridge=1e-6, prior_alpha=0.):
    return strict_online_discounted_ols_gate_mix(target, checkpoint, proposal,
        segment_len=segment_len, warmup=warmup, half_life_rows=half_life_rows,
        ridge=ridge, prior_alpha=prior_alpha, feedback_delay=0)
