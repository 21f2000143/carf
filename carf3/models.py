"""Model registry for the CARF-2 study (all fusions receive the same expert outputs)."""
from fusion import Mean, Stack, MaskStack, OffsetLR, ModDropMLP, CARFWrap, Gated

BUDGETS = (1.0, 1.5, 2.0, 3.0)
BASELINES = ['mean', 'stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'moddrop_mlp']
CARF = ['carf_v1', 'carf2']
GATED = ['gated_' + b for b in BASELINES]
ABLATIONS = ['carf2_no_gate', 'carf2_no_robust', 'carf2_no_swap', 'carf2_B1', 'carf2_const', 'carf2_unbounded']
GATE = dict(gate_power=1.0, gate='posterior', prior=0.95)
SX = ('swap', 'extreme')


def make(name):
    if name.startswith('gated_'):
        return Gated(make(name[len('gated_'):]), prior=GATE['prior'])
    return {
        'mean': lambda: Mean(),
        'stack': lambda: Stack(),
        'mask_stack': lambda: MaskStack(),
        'offset_lr': lambda: OffsetLR(),
        'robust_offset_lr': lambda: OffsetLR(robust_weight=.25),
        'moddrop_mlp': lambda: ModDropMLP(),
        # Original CARF: B = 1, extreme corruptions only.
        'carf_v1': lambda: CARFWrap(budget=1.0),
        # CARF-2: anchor-match gate on every optional residual, sample-swap + extreme environments,
        # budget chosen by inner CV.
        'carf2': lambda: CARFWrap(cv_budgets=BUDGETS, envs=SX, **GATE),
        'carf2_no_gate': lambda: CARFWrap(cv_budgets=BUDGETS, envs=SX),
        'carf2_no_robust': lambda: CARFWrap(cv_budgets=BUDGETS, robust_weight=0., **GATE),
        'carf2_no_swap': lambda: CARFWrap(cv_budgets=BUDGETS, envs=('extreme',), **GATE),
        'carf2_B1': lambda: CARFWrap(budget=1.0, envs=SX, **GATE),
        'carf2_const': lambda: CARFWrap(cv_budgets=BUDGETS, envs=SX, bases=('const',), **GATE),
        'carf2_unbounded': lambda: CARFWrap(budget=1.0, envs=SX, bounded=False, **GATE),
    }[name]()


ALL = BASELINES + CARF + ABLATIONS
