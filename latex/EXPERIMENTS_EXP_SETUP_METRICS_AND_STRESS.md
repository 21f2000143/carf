Yes — this paragraph is much easier once you realize that it is describing **two completely different things**:

1. **How they measure normal model performance**
2. **How they deliberately attack the trained models to see whether CARF is robust**

The second part is the **stress test**.

And yes, **for a paper whose main claim is robustness to missing/corrupted modalities, some form of stress testing is essentially necessary**. But you do **not** necessarily need every stress test they used.

Let's go from zero.

---

# 1. First: what are they trying to prove?

From everything we've discussed, CARF is **not mainly claiming**:

> "I predict better than every other model."

Its important claim is:

> **"If an optional modality becomes unreliable, CARF should be more stable than ordinary fusion methods."**

For example:

```text
Normal patient

Clinical → 0.70
RNA      → 0.80
MRI      → 0.65

             ↓
            CARF
             ↓
          prediction
```

Now imagine RNA becomes garbage:

```text
Clinical → 0.70
RNA      → 0.01  ← corrupted
MRI      → 0.65

             ↓
            CARF
             ↓
       should remain
       reasonably stable
```

How can the authors demonstrate this claim?

They need to **deliberately corrupt RNA/MRI and see what happens**.

That's the stress test.

---

# 2. Normal evaluation vs stress testing

This distinction is crucial.

### Normal evaluation

Ask:

> "How good is the model when the data are normal?"

```text
Clean patient
     ↓
Model
     ↓
AUROC
```

### Stress testing

Ask:

> "What happens when I deliberately damage one modality?"

```text
Clean patient
     ↓
Damage RNA
     ↓
Model
     ↓
How much did performance fall?
```

So:

$$
\boxed{\text{Normal test = accuracy}}
$$

while:

$$
\boxed{\text{Stress test = robustness}}
$$

---

# 3. AUROC

They report **AUROC**.

You've probably already encountered this.

Very roughly:

> AUROC measures how well the model separates positive and negative patients.

Higher is generally better.

For example:

```text
CARF       AUROC = 0.82
Stacking   AUROC = 0.84
Clinical   AUROC = 0.80
```

This tells you about discrimination.

But it doesn't tell you everything.

---

# 4. Brier score

Brier score asks a different question:

> **Are the probabilities themselves sensible?**

Suppose the model says:

```text
Patient A → 0.90 probability of disease
```

and disease actually happens.

That's a good probabilistic prediction.

But suppose:

```text
Patient B → 0.90
```

and disease doesn't happen.

That's a bad probabilistic prediction.

Brier score evaluates these probability errors.

**Lower Brier score is better.**

So:

* AUROC → **Can the model rank patients correctly?**
* Brier → **Are the predicted probabilities numerically reasonable?**

---

# 5. ECE

ECE = **Expected Calibration Error**.

This is another way of asking:

> **When the model says 80%, does the event actually happen about 80% of the time?**

Imagine 100 patients for whom the model predicts approximately 80%.

If approximately 80 of them actually have the outcome:

```text
Predicted = 80%
Observed  = 80%
```

That's well calibrated.

If only 50 actually have the outcome:

```text
Predicted = 80%
Observed  = 50%
```

that's poorly calibrated.

So:

> **Calibration = whether predicted probabilities correspond to actual frequencies.**

Again, you don't need to memorize the mathematical formula yet.

---

# 6. "Paired bootstrap intervals"

This sounds terrible but the idea is simple.

Suppose they compare:

```text
CARF AUROC = 0.82
Stacking AUROC = 0.84
```

Difference:

$$
0.82-0.84=-0.02
$$

But is that difference real or could it just be random variation from the particular patients?

They use **bootstrap resampling** to estimate uncertainty around the difference.

Imagine 1,000 patients.

They repeatedly sample patients **with replacement**:

```text
Original:
1 2 3 4 5 ... 1000

Bootstrap sample:
5 5 8 10 10 13 ... 997
```

Then calculate the AUROC difference.

Repeat this many times.

Here they use:

$$
2000
$$

resamples.

Eventually they get a distribution of possible AUROC differences.

From that they calculate an interval.

So:

> **Bootstrap interval = an estimate of how uncertain the observed performance difference is.**

---

# 7. Now we reach the important sentence

> "Stress tests freeze all models and alter one optional modality of the external patients"

**Freeze all models** means:

> They do NOT retrain the models after corrupting the data.

This is extremely important.

Suppose they trained CARF:

```text
I-SPY2
 ↓
train CARF
 ↓
CARF is now frozen
```

Then take I-SPY1:

```text
I-SPY1
 ↓
clean data
 ↓
CARF
```

Then they corrupt RNA:

```text
I-SPY1
 ↓
RNA corrupted
 ↓
SAME CARF
 ↓
prediction
```

They don't allow CARF to learn from the corrupted data.

Why?

Because they want to test:

> **What happens to the already-trained model when reality suddenly becomes messy?**

---

# 8. Why "one optional modality"?

Remember:

```text
Clinical = anchor
RNA = optional
MRI = optional
```

They don't want to destroy everything simultaneously.

Instead:

### Experiment A

```text
Clinical = normal
RNA = corrupted
MRI = normal
```

### Experiment B

```text
Clinical = normal
RNA = normal
MRI = corrupted
```

This tells them specifically how the model handles failure of each optional modality.

---

# 9. "with 30 seeds each"

A **seed** controls randomness.

Suppose they randomly corrupt RNA.

You could randomly select different patients/genes each time.

One random experiment:

```text
Seed 1
```

Another:

```text
Seed 2
```

...

up to:

```text
Seed 30
```

So they repeat the corruption experiment 30 different ways.

Why?

Because otherwise you might accidentally get a particularly mild or severe corruption.

So:

> **30 seeds = repeat the random corruption 30 times to make the result less dependent on one random realization.**

---

# 10. Output-level corruption

This is probably the most important stress test to understand.

They say:

> "At the output level, we permute the expert probability across patients"

Suppose RNA expert gives:

| Patient | RNA probability |
| ------- | --------------: |
| A       |            0.90 |
| B       |            0.80 |
| C       |            0.20 |
| D       |            0.10 |

Normally:

```text
A → 0.90
B → 0.80
C → 0.20
D → 0.10
```

Now **permute** them:

| Patient | Original | After permutation |
| ------- | -------: | ----------------: |
| A       |     0.90 |              0.10 |
| B       |     0.80 |              0.20 |
| C       |     0.20 |              0.90 |
| D       |     0.10 |              0.80 |

The RNA probabilities still have roughly the same distribution.

But they are now attached to the **wrong patients**.

So RNA becomes essentially **uninformative about the individual patient**.

---

# 11. Why is this a good corruption?

Because they want to simulate:

> **The RNA modality has become useless, while still looking numerically normal.**

That's clever.

They aren't simply replacing RNA with:

```text
0, 0, 0, 0...
```

because that might create an unrealistic input.

Instead:

```text
Patient A → RNA prediction belonging to Patient C
Patient B → RNA prediction belonging to Patient D
...
```

The numbers still look normal.

But the patient-specific information has been destroyed.

---

# 12. Why do they call it "not used in training"?

This is important.

They trained the model using certain corruptions.

Then, during testing, they use permutation corruption that the model **didn't see during training**.

That tests whether CARF can handle a type of failure it wasn't specifically trained against.

So:

```text
Training:
missingness
Gaussian noise
other corruptions

Testing:
PERMUTATION
```

That's a stronger robustness test than simply testing exactly the corruption the model was trained on.

---

# 13. Input-level corruption

Now instead of corrupting the **expert's output probability**, they corrupt the **original input data**.

This is the distinction:

### Output-level

```text
RNA data
   ↓
RNA model
   ↓
0.82
   ↓
CORRUPT 0.82
```

### Input-level

```text
RNA data
   ↓
CORRUPT RNA data
   ↓
RNA model
   ↓
new probability
```

So input-level corruption happens earlier.

---

# 14. "Remove 50% of the genes"

Suppose one patient has:

```text
20,000 genes
```

They randomly remove 50%:

```text
20,000
   ↓
10,000 genes remain
```

Then run the model.

This simulates:

> **What if a large fraction of transcriptomic measurements are unavailable?**

---

# 15. "Shift 25% of the genes by two log-ratio units"

This means they deliberately change the expression values of 25% of the genes.

For example, conceptually:

```text
Normal:
Gene A = 4.2

Corrupted:
Gene A = 6.2
```

for the selected genes, according to their specified log-ratio transformation.

The important idea is:

> **Some gene measurements are systematically shifted.**

This simulates something like a distribution/platform/batch change.

---

# 16. MRI noise

They also corrupt MRI features.

> "add noise of one training SD to the MRI features"

**SD = standard deviation.**

Suppose a particular MRI feature normally has:

$$
SD=2
$$

They add random noise with that scale.

So:

```text
Original MRI feature = 10
Noise = +1.7

Corrupted = 11.7
```

Another patient might get:

```text
Original = 8
Noise = -2.1

Corrupted = 5.9
```

So they are asking:

> What happens if MRI measurements become noisy?

---

# 17. "remove 60% of them"

This refers to MRI features.

If there are 100 MRI features:

```text
100
 ↓
remove 60
 ↓
40 remain
```

Again:

> What happens when a large amount of MRI information disappears?

---

# 18. What are "stress intervals"?

They repeat these corruption experiments.

For example:

```text
Seed 1 → AUROC = 0.78
Seed 2 → AUROC = 0.75
Seed 3 → AUROC = 0.79
...
Seed 30 → AUROC = 0.76
```

So they have 30 results.

They calculate an interval around the stress-test performance.

That's the **stress interval**.

---

# 19. "resample patients with the 30 realizations held fixed"

This sentence is very technical.

It basically means:

> First create the 30 corruption versions.

```text
Corruption 1
Corruption 2
...
Corruption 30
```

Then when they bootstrap the patients, they **don't generate new corruption every time**.

They keep those 30 corruption versions fixed.

So they separate two sources of randomness:

1. randomness from **which corruption happened**
2. randomness from **which patients are resampled**

This makes their uncertainty calculation cleaner.

---

# 20. Finally: the permutation test

This last sentence is actually a **different analysis** from the stress test.

> "Refitting the whole pipeline on permuted I-SPY2 labels yields p = 0.001"

They want to know:

> **Could CARF's external performance simply be a lucky result caused by random labels?**

So they destroy the relationship between patient data and outcome.

Suppose original labels are:

```text
Patient A → Cancer
Patient B → No cancer
Patient C → Cancer
Patient D → No cancer
```

They randomly shuffle them:

```text
Patient A → No cancer
Patient B → Cancer
Patient C → No cancer
Patient D → Cancer
```

Now the labels are unrelated to the patients' actual features.

---

# 21. Then they repeat the entire experiment

This is important.

They don't just shuffle labels and test the already-trained CARF.

They say:

> **"Refitting the whole pipeline"**

Meaning:

```text
Shuffle labels
      ↓
Train everything again
      ↓
Select/tune again
      ↓
Train CARF again
      ↓
Evaluate on external data
```

Then repeat this **1,000 times**.

They ask:

> How often can completely random labels produce an external AUROC at least as large as the observed one?

They obtained:

$$
p=0.001
$$

---

# 22. Why do they say "smallest value attainable with 1000 permutations"?

This is just an important statistical detail.

With 1,000 permutations, the smallest non-zero p-value is approximately:

$$
\frac{1}{1000}=0.001
$$

depending on the exact permutation-test convention.

So if none of the 1,000 random-label experiments produced performance as high as the real experiment, they report:

$$
p=0.001
$$

rather than claiming:

$$
p=0
$$

There is no such thing as evidence of literally zero probability from a finite permutation experiment.

---

# 23. Now the whole paragraph in plain English

Here is what the authors are actually saying:

> **We measured normal model performance using AUROC and probability-quality measures such as Brier score and calibration. We also estimated uncertainty in the differences between models.**
>
> **Then we tested robustness. We froze the trained models and deliberately damaged one optional modality at a time in the independent test patients. We repeated each random corruption 30 times.**
>
> **We tested corruption both after the expert had produced its probability and earlier, directly on the original RNA/MRI data.**
>
> **For example, we scrambled RNA predictions between patients, removed many genes, shifted some gene measurements, added noise to MRI features, or removed many MRI features.**
>
> **Then we measured how much each model's performance deteriorated.**
>
> **Finally, we shuffled the training labels 1,000 times and repeated the entire training pipeline to check whether CARF's external performance could plausibly arise from random labels.**

That's it.

---

# 24. Now your important question: **Is stress testing actually required?**

For **this particular research claim**, I would say **yes, some stress testing is necessary**.

Why?

Because suppose the paper only reported:

```text
Normal I-SPY1 AUROC

CARF       0.80
Stacking   0.82
Clinical   0.78
```

What does that prove about CARF?

Almost nothing about robustness.

It only tells us how the models behave on the normal test dataset.

The central claim is about **corrupted/missing modalities**.

Therefore you need something like:

```text
                    Normal       RNA corrupted
CARF                 0.80           0.77
Stacking             0.82           0.60
```

Now you can actually demonstrate:

> When RNA becomes unreliable, CARF loses less performance.

That's the evidence supporting the robustness claim.

---

# 25. But you do NOT necessarily need every corruption

This is where I would simplify the experimental design.

The authors have a fairly large collection:

### RNA

* permutation
* remove 50% genes
* shift 25% genes
* Gaussian noise
* etc.

### MRI

* Gaussian noise
* remove 60% features
* permutation
* etc.

You don't necessarily need 10 different attacks.

A clean research design could have:

### 1. Normal condition

```text
No corruption
```

### 2. Missing modality

```text
RNA missing
MRI missing
```

### 3. Uninformative modality

```text
RNA prediction permutation
MRI prediction permutation
```

### 4. Measurement degradation

```text
RNA feature noise / gene removal
MRI feature noise / feature removal
```

That's already a strong robustness evaluation.

---

# 26. The most important distinction for your understanding

Remember these three levels:

### Level 1 — Normal performance

> **Is the model accurate?**

```text
Clean data → model → AUROC
```

### Level 2 — Stress test

> **Does the model survive when a modality becomes unreliable?**

```text
Corrupted data → frozen model → AUROC
```

### Level 3 — Permutation test

> **Could the apparent predictive performance arise from random labels?**

```text
Random labels
     ↓
retrain everything
     ↓
external AUROC
```

They answer **three different questions**.

---

## And the most important thing for CARF

The entire robustness experiment can be summarized as:

```text
                    TRAIN ON I-SPY2
                           │
                           ▼
                    ┌────────────┐
                    │ CARF       │
                    │ STACKING   │
                    │ etc.       │
                    └─────┬──────┘
                          │
                       FREEZE
                          │
                          ▼
                     I-SPY1
                          │
                ┌─────────┴─────────┐
                │                   │
             CLEAN              CORRUPTED
                │                   │
                ▼                   ▼
             AUROC              AUROC
                │                   │
                └─────────┬─────────┘
                          ▼
                  How much did
                  performance fall?
```

**That last question is the heart of the stress test.**

If CARF's whole purpose is *"optional modalities may become missing or corrupted, but the anchor should remain protected,"* then showing **how much performance degrades under controlled corruption** is not an optional decoration—it is the empirical test of the central claim.
