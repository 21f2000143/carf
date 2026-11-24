"""Architecture diagram of CARF (anchor-verified bounded fusion) for latex2/figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

FIG = Path(__file__).resolve().parents[1] / 'latex2/figures'
CM = 1 / 2.54
plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42})


def main():
    W, H = 12.2, 5.6
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fs, fs2 = 6.8, 6.2

    def box(x, y, w, h, text='', fc='#FFFFFF', ec='#333333', lw=0.7, ls='-', fsz=fs, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=lw, ls=ls))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fsz, **kw)

    def arrow(p, q, color='#333333', ls='-', lw=0.8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=6, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))

    CLIN, RNA, MRI, RPPA, GATE = '#E8E8E8', '#DCEBF7', '#FBE3D2', '#E3F1DF', '#FFF4C8'
    rows = [('RNA', 'R', RNA), ('MRI', 'M', MRI), ('RPPA', 'P', RPPA)]
    yC = 4.95; ys = [4.1, 3.35, 2.6]; bh = 0.5
    x1, w1 = 0.05, 1.25     # experts
    x2, w2 = 1.6, 1.95      # residual
    x3, w3 = 3.75, 0.75      # clip
    xg, wg = 4.85, 1.1      # gate
    x4, w4 = 6.35, 2.0      # sum
    box(x1, yC - bh / 2, w1, bh, 'Clinical', fc=CLIN)
    box(x2, yC - bh / 2, w2, bh, r'$a=\mathrm{logit}\,p_C$', fc=CLIN, lw=1.1)
    arrow((x1 + w1, yC), (x2, yC))
    ax.text(x1 + w1 / 2, ys[0] + bh / 2 + 0.03, 'optional', ha='center', va='bottom', fontsize=fs2, style='italic')
    for (name, s, fc), yy in zip(rows, ys):
        box(x1, yy - bh / 2, w1, bh, name, fc=fc)
        box(x2, yy - bh / 2, w2, bh, rf'$d_{s}=\mathrm{{logit}}\,p_{s}-a$', fsz=6.4, fc=fc)
        arrow((x1 + w1, yy), (x2, yy))
        box(x3, yy - bh / 2, w3, bh, fc=fc)
        ix, iy, sz = x3 + 0.08, yy - 0.15, 0.3
        ax.plot([ix, ix + sz * .3, ix + sz * .7, ix + sz], [iy, iy, iy + sz, iy + sz], color='#333', lw=0.8)
        ax.text(x3 + 0.58, yy, r'$r_j$', ha='center', va='center', fontsize=fs)
        arrow((x2 + w2, yy), (x3, yy))
        box(xg, yy - bh / 2, wg, bh, r'$\times\, g_j$', fc=GATE)
        arrow((x3 + w3, yy), (xg, yy))
        arrow((xg + wg, yy), (x4, yy))
    ax.text(x3 + w3 / 2, ys[0] + bh / 2 + 0.03, r'clip $\pm B$', ha='center', va='bottom', fontsize=fs2)
    ax.plot([x2 + w2 + 0.1] * 2, [ys[0] + bh / 2, ys[-1] - bh / 2], color='#777', lw=0.6, ls=(0, (2, 1.5)))
    y4a, y4b = ys[-1] - bh / 2, ys[0] + bh / 2
    box(x4, y4a, w4, y4b - y4a, lw=0.9)
    cx = x4 + w4 / 2
    ax.text(cx, (y4a + y4b) / 2, r'$\sum_{j,k}\theta_{jk}\,m_j\,g_j\,h_{jk}\,r_j$', ha='center', va='center', fontsize=7.2)
    ax.add_patch(Circle((cx, yC), 0.18, fc='white', ec='#333', lw=0.8))
    ax.text(cx, yC - 0.01, '+', ha='center', va='center', fontsize=8.5)
    arrow((x2 + w2, yC), (cx - 0.18, yC)); arrow((cx, y4b), (cx, yC - 0.18))
    box(cx + 0.45, yC - 0.2, 0.4, 0.4, r'$\sigma$')
    arrow((cx + 0.18, yC), (cx + 0.45, yC)); ax.text(cx + 0.31, yC + 0.04, '$z$', ha='center', va='bottom', fontsize=fs2)
    arrow((cx + 0.85, yC), (cx + 1.1, yC)); ax.text(cx + 1.15, yC, r'$\hat p$', ha='left', va='center', fontsize=8.5)
    tx = x4 + w4 + 0.15
    ax.text(tx, y4b + 0.02, r'$h_{j0}=1,\ h_{j1}=e^{-|d_j|}$', ha='left', va='top', fontsize=fs2)
    ax.text(tx, y4b - 0.33, r'$h_{j2}=1-H(p_j)/\log 2$', ha='left', va='top', fontsize=fs2)
    ax.text(tx, y4b - 0.78, r'$g_j=\Pr(\mathrm{match}\mid$', ha='left', va='top', fontsize=fs2)
    ax.text(tx + 0.2, y4b - 1.1, r'aux. HR/HER2, record$)$', ha='left', va='top', fontsize=fs2)
    ax.text(tx, y4a + 0.02, r'$\theta\geq0,\ \sum\theta\leq1\Rightarrow|z-a|\leq B$', ha='left', va='bottom', fontsize=fs2)
    # training strip
    yg0, yg1 = 0.03, 1.5
    box(0.05, yg0, W - 0.1, yg1 - yg0, fc='#F5F5F5', ec='#888', lw=0.7, ls=(0, (3, 2)))
    gx, gy, dy = 0.2, yg1 - 0.08, 0.33
    ax.text(gx, gy, 'Gate fitting (convex in $\\theta$)', ha='left', va='top', fontsize=fs, fontweight='bold')
    ax.text(W - 0.2, gy, r'$\gamma=0.25,\ \tau=0.05,\ \lambda=0.01$; $B$ by inner CV', ha='right', va='top', fontsize=fs2, color='#444')
    ax.text(gx, gy - dy, r'Inner out-of-fold expert outputs under all $2^K$ availability masks of the optional modalities,',
            ha='left', va='top', fontsize=fs2)
    ax.text(gx, gy - 2 * dy, r'plus failures: sample swaps (gate re-checked) and extreme outputs (0.01 / 0.99)',
            ha='left', va='top', fontsize=fs2)
    ax.text(gx, gy - 3 * dy, r'Loss: mean log loss (masks) $+\,\gamma\cdot$smooth-max (failures) $+\,\lambda\|\theta\|^2$',
            ha='left', va='top', fontsize=fs2)
    arrow((cx, yg1), (cx, y4a - 0.0), color='#888', ls=(0, (3, 2)))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'fig_architecture.pdf'); fig.savefig(FIG / 'fig_architecture.png', dpi=220)


if __name__ == '__main__':
    main()
