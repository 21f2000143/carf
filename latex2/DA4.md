# can only shrink the residuals, so it does not weaken any bound" I did not understand where this b is being used in the entire paper, it is not there in z_\theta CARF equation?

Exactly — **\(b\) is not an additional term in the CARF prediction equation.** It is a **derived quantity used to prove safety guarantees**.

This is the key distinction.

### 1. The actual CARF prediction equation

The model is:

$$
z_\theta
=
a+
\sum_j\sum_k
\theta_{jk}m_jg_jh_{jk}r_j
$$

There is no \(b\) here.

Here:

* \(a\) = clinical anchor logit
* \(r_j\) = clipped residual
* \(h_{jk}\) = context scaling
* \(g_j\) = verification gate
* \(\theta_{jk}\) = learned weights
* \(m_j\) = missingness mask

So **\(b\) is not used to calculate the prediction directly.**

---

# 2. Then where does \(b\) come from?

They define:

$$
b_j = B\sum_k\theta_{jk}
$$

and

$$
b=\sum_j b_j
$$

This is simply a **mathematical upper bound on how much all optional modalities together can move the anchor**.

Remember:

$$
|r_j|\leq B
$$

and:

$$
0\leq g_j,h_{jk}\leq1
$$

Therefore:

$$
|m_jg_jh_{jk}r_j|\leq B
$$

So each contribution satisfies:

$$
|\theta_{jk}m_jg_jh_{jk}r_j|
\leq
\theta_{jk}B
$$

Adding everything:

$$
|z_\theta-a|
\leq
B\sum_j\sum_k\theta_{jk}
$$

They define that right-hand side as:

$$
\boxed{b=B\sum_{j,k}\theta_{jk}}
$$

That's where \(b\) comes from.

---

# 3. Think of \(b\) as a "maximum allowed movement"

Suppose:

$$
B=1
$$

and the learned coefficients are:

| Modality | \(\theta_{j0}\) | \(\theta_{j1}\) | \(\theta_{j2}\) |
| -------- | --------------: | --------------: | --------------: |
| RNA      |            0.20 |            0.10 |            0.10 |
| MRI      |            0.15 |            0.10 |            0.05 |
| RPPA     |            0.10 |            0.05 |            0.05 |

Then:

$$
\sum_{j,k}\theta_{jk}=1
$$

so:

$$
b=B(1)=1
$$

That means:

$$
\boxed{|z_\theta-a|\leq1}
$$

The model can move the clinical logit by at most 1.

But **the model doesn't calculate**

$$
z_\theta=a+1
$$

or anything like that.

It calculates the original equation normally.

The \(1\) is simply a **guaranteed maximum**.

---

# 4. Actual prediction vs guarantee

This distinction is extremely important.

Suppose:

$$
a=1.2
$$

and the actual residual contributions happen to be:

$$
RNA=+0.15
$$

$$
MRI=-0.08
$$

$$
RPPA=+0.05
$$

Then:

$$
z_\theta=1.2+0.15-0.08+0.05
$$

$$
z_\theta=1.32
$$

The actual displacement is:

$$
|1.32-1.2|=0.12
$$

But CARF may have:

$$
b=1
$$

Therefore it guarantees:

$$
0.12\leq1
$$

So:

```text
Actual:
anchor ───────────────► final
  1.20                  1.32
       displacement = 0.12

Guaranteed:
maximum displacement = b = 1
```

---

# 5. Why is this useful?

Because the authors want to make a **robustness claim**.

They can say:

> Regardless of what the optional modalities produce, the final prediction cannot move more than \(b\) logits away from the clinical anchor.

That's much stronger than simply saying:

> "Our model usually behaves well."

It's a **mathematical guarantee**.

---

# 6. Why does the paper say "budget"?

Because you can think of the \(\theta\)'s as distributing a limited amount of influence.

They impose:

$$
\sum_{j,k}\theta_{jk}\leq1
$$

and because:

$$
|r_j|\leq B
$$

the total possible movement is bounded by:

$$
b\leq B
$$

So if:

$$
B=1
$$

then the entire multimodal system has at most **1 logit unit of correction budget**.

The modalities effectively "spend" that budget.

---

# 7. What is \(b_j\)?

They also divide the total budget among modalities:

$$
b_j=B\sum_k\theta_{jk}
$$

For example:

```text
Total budget b = 1.0

RNA      → 0.40
MRI      → 0.30
RPPA     → 0.20
Other    → 0.10
          ─────
           1.00
```

So RNA can potentially cause at most \(0.40\) logit displacement, MRI \(0.30\), etc.

This lets them prove another property:

> If you replace/remove/swap **only MRI**, the prediction can change by at most \(2b_{\text{MRI}}\).

Why \(2\)?

Because MRI could go from its maximum positive contribution:

$$
+b_j
$$

to its maximum negative contribution:

$$
-b_j
$$

so the total possible change is:

$$
+b_j-(-b_j)=2b_j
$$

---

# 8. The \(b/4\) probability bound

They also state:

$$
|\hat p-p_C|\leq \frac b4
$$

Why?

Because the sigmoid function has maximum slope:

$$
\sigma'(z)\leq\frac14
$$

Therefore a logit change of at most \(b\) produces a probability change of at most:

$$
\frac b4
$$

For example, if:

$$
b=0.88
$$

then:

$$
|\hat p-p_C|\leq0.22
$$

This is a **worst-case mathematical bound**, not necessarily the actual observed difference.

---

# 9. So the whole concept is

```text
              ACTUAL MODEL
                    │
                    ▼
        z = a + Σ θ·m·g·h·r
                    │
                    │
                    ▼
              actual prediction


              MATHEMATICAL ANALYSIS
                    │
                    ▼
       |θ·m·g·h·r| ≤ θB
                    │
                    ▼
             define b
                    │
                    ▼
       |z − a| ≤ b
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 probability bound      ranking guarantee
 |p − pC| ≤ b/4         anchor ordering preserved
```

### The simplest answer to your question:

**\(b\) is not another component of CARF. It is a quantity derived from \(B\) and the learned \(\theta\)'s to quantify the maximum influence that the optional modalities are allowed to have.**

So when you see:

$$
\boxed{|z_\theta-a|\leq b}
$$

read it as:

> **"Whatever the RNA/MRI/RPPA values are, CARF guarantees that their combined effect cannot move the clinical anchor by more than \(b\)."**

That's why \(b\) doesn't appear in the original \(z_\theta\) equation.
