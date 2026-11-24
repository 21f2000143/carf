Absolutely. Let's forget the complicated ML terminology and understand this **like a simple experiment with students**.

The paragraph is describing **how they tested CARF fairly**.

---

## 1. First, what is "I-SPY2"?

Think of **I-SPY2** as their main dataset.

It contains patients with information such as:

* clinical information
* RNA/transcriptomic information
* MRI information
* outcome/label

They use this dataset to **develop and test** their models.

**I-SPY1** is a separate dataset.

They use I-SPY1 only at the end to see how the already-developed models behave on completely separate data.

So:

```text
I-SPY2
   │
   ├── Develop/test the method
   │
   └── Cross-validation
       
I-SPY1
   │
   └── Final external test
```

---

# 2. What is "training a model"?

Suppose we have 100 patients.

For each patient:

```text
RNA + clinical + MRI  →  disease outcome
```

We want the computer to learn a relationship between the input and outcome.

For example:

```text
Clinical information
       +
RNA
       +
MRI
       ↓
     MODEL
       ↓
Predicted probability
```

**Training** simply means:

> Give the model examples where we know the correct answer, so it can learn how to make predictions.

---

# 3. What is "evaluation"?

After training, we give the model patients it **didn't use for learning**.

For example:

```text
Training patients
100 patients
     ↓
  MODEL LEARNS

New patients
20 patients
     ↓
MODEL MAKES PREDICTIONS
```

Then we compare:

```text
Prediction vs actual outcome
```

to calculate AUROC or another performance measure.

This is important because if you test the model on the exact same patients it learned from, you can get an overly optimistic result.

---

# 4. What is "cross-validation"?

This sounds complicated but the idea is extremely simple.

Suppose I have **10 patients**.

Instead of:

```text
Train on 10
Test on 10
```

I split them into groups.

For example:

```text
Group 1: patients 1–2
Group 2: patients 3–4
Group 3: patients 5–6
Group 4: patients 7–8
Group 5: patients 9–10
```

Then I do:

### Round 1

```text
TRAIN:  3 4 5  → etc.
TEST:   Group 1
```

### Round 2

```text
TRAIN:  everything except Group 2
TEST:   Group 2
```

and so on.

Eventually every patient has been used as a test patient.

This is called:

> **5-fold cross-validation**

because we divide the data into **5 groups (folds)**.

---

# 5. What does "stratified" mean?

Suppose our dataset contains:

```text
80 patients without event
20 patients with event
```

We don't want one fold to accidentally contain:

```text
Fold 1 → 20 events
Fold 2 → 0 events
Fold 3 → 0 events
...
```

Instead, we try to preserve approximately the same proportion in every fold.

For example:

```text
Fold 1 → 16% events
Fold 2 → 20% events
Fold 3 → 18% events
Fold 4 → 21% events
Fold 5 → 19% events
```

That's **stratification**.

So:

> **Stratified = keep the important class proportions approximately similar in each group.**

---

# 6. What does "five repeats of stratified five-fold cross-validation" mean?

This is simply:

### First repetition

```text
5-fold CV
```

### Second repetition

They randomly divide the data into five folds again:

```text
5-fold CV
```

### Third

```text
5-fold CV
```

### Fourth

```text
5-fold CV
```

### Fifth

```text
5-fold CV
```

So:

$$
5\text{ repeats} \times 5\text{ folds}
=25\text{ evaluations}
$$

That's why later they say:

> "below it in 20 of 25 folds"

There are **25 test rounds**.

---

# 7. Now the important phrase: "outer training set"

This is where the paragraph gets harder.

They aren't actually doing just ordinary 5-fold cross-validation.

They're doing **nested cross-validation**.

Don't worry about the name.

Think of it as:

> **A big exam and a small practice exam inside it.**

Suppose:

```text
100 patients
```

The outer cross-validation says:

```text
80 patients → development/training
20 patients → final test
```

The 20 patients must be kept completely untouched.

Now we take the 80 patients:

```text
80 patients
    ↓
inner cross-validation
    ↓
used to make decisions about the model
```

So:

```text
                 100 patients
                      │
             ┌────────┴────────┐
             │                 │
          80 patients       20 patients
          OUTER TRAIN       OUTER TEST
             │                 │
             │                 │
       inner CV              untouched
             │
             ↓
       choose/fit things
```

---

# 8. Why do they need the inner cross-validation?

This is especially important for CARF.

Remember that CARF has different models/components that need to be fitted.

If you simply train everything on the 80 outer-training patients, you might accidentally use information from the same patients to both:

> create predictions that train the fusion model

and

> train the fusion model itself.

That can cause **information leakage**.

---

# 9. What is "expert prediction"?

This is another term you need to understand.

Suppose they have:

### Clinical expert

```text
Clinical data → Clinical model → 0.70
```

### RNA expert

```text
RNA data → RNA model → 0.80
```

### MRI expert

```text
MRI data → MRI model → 0.65
```

These three models are called **experts**.

Their outputs are:

```text
Clinical expert → 0.70
RNA expert      → 0.80
MRI expert      → 0.65
```

Then CARF takes these outputs and combines them.

So:

```text
Clinical ──→ Clinical expert ──→ 0.70 ──┐
                                        │
RNA ──────→ RNA expert ────────→ 0.80 ──┤→ CARF → final prediction
                                        │
MRI ──────→ MRI expert ────────→ 0.65 ──┘
```

The **fusion model** is the thing that combines these expert predictions.

---

# 10. So what does "inner cross-fitting produces the expert predictions" mean?

This is probably the most confusing sentence.

Imagine our outer training data contains:

```text
80 patients
```

They don't want to simply do:

```text
80 patients
   ↓
train RNA expert
   ↓
predict same 80 patients
```

because the RNA model has already seen those patients.

Instead, they use **inner cross-validation**.

For example:

```text
80 patients
    ↓
divide into 5 groups
```

Then:

```text
Train RNA model on 64 patients
             ↓
       predict 16 patients
```

Then another group:

```text
Train RNA model on another 64
             ↓
       predict the remaining 16
```

Eventually every one of the 80 patients receives a prediction from an RNA model that **did not train on that particular patient**.

These are called:

> **out-of-fold predictions**

or, in this paper's language:

> **cross-fitted expert predictions**.

---

# 11. Why is this necessary?

This is extremely important.

Suppose patient A's true outcome is:

```text
Cancer = YES
```

If the RNA expert was trained using patient A and then predicts patient A:

```text
RNA expert
    ↑
already saw patient A
    ↓
predicts patient A
```

The prediction may be unrealistically good.

Then CARF learns from this artificially good prediction.

That's unfair.

Instead:

```text
Patient A
   ↓
RNA expert trained WITHOUT patient A
   ↓
prediction for patient A
```

Now the prediction behaves more like what the model would produce for a genuinely new patient.

That's **cross-fitting**.

---

# 12. Now let's translate the entire first sentence

Original:

> "Internal evaluation uses five repeats of stratified five-fold cross-validation on I-SPY2"

Simple meaning:

> **They repeatedly divided I-SPY2 into five groups, trained on four groups and tested on the remaining group, doing this five times with different splits.**

---

# 13. Next sentence

Original:

> "in each outer training set, inner cross-fitting produces the expert predictions used to fit every fusion model"

Break it apart.

### "outer training set"

The large portion of data used for model development.

```text
I-SPY2
 ↓
80% outer training
20% outer testing
```

### "inner cross-fitting"

Inside that 80%:

```text
80%
 ↓
inner CV
```

They make predictions for each patient using an expert that **didn't train on that patient**.

### "expert predictions"

For example:

```text
Patient 1:
Clinical expert = 0.71
RNA expert      = 0.82
MRI expert      = 0.65
```

### "used to fit every fusion model"

Those numbers become the input to:

```text
Stacking
CARF
other fusion models
```

So:

```text
RAW DATA
   │
   ├── Clinical → expert → prediction
   │
   ├── RNA      → expert → prediction
   │
   └── MRI      → expert → prediction
                         │
                         ↓
                 Fusion model
               ┌───────────────┐
               │ CARF          │
               │ Stacking      │
               │ Other models  │
               └───────────────┘
                         │
                         ↓
                  Final prediction
```

---

# 14. What does "external models are fitted in the same way on all of I-SPY2" mean?

This one is easier.

For the **external test**, they use **all of I-SPY2** to train the final models.

Why?

Because now they're finished developing the method.

So:

```text
During internal evaluation:

I-SPY2
 ↓
train/test repeatedly
```

After that process is finished:

```text
ALL I-SPY2
     ↓
train final model
     ↓
take to I-SPY1
```

The idea is:

> Use all available development data to train the final version before testing it on a completely separate dataset.

---

# 15. Why is I-SPY1 called "external"?

Because it was not part of the training data.

Think about a student.

### I-SPY2

```text
Practice + mock exams
```

### I-SPY1

```text
REAL final exam
```

You don't look at the answers to the final exam while studying.

Similarly:

```text
I-SPY2 → model development
I-SPY1 → independent evaluation
```

---

# 16. The last sentence is VERY important

> "I-SPY1 labels are used for neither fitting nor hyperparameter choice."

What is a **label**?

The label is the correct answer.

For example:

```text
Patient
   ↓
Input:
RNA + MRI + clinical
   ↓
True outcome:
YES
```

"YES" is the **label**.

---

# 17. What is "fitting"?

Fitting = training the model.

So:

> I-SPY1 labels were not used for training.

Good.

---

# 18. What is a "hyperparameter"?

This is another ML term.

A normal model **learns parameters** from data.

For example:

```text
Model learns:
weight1 = 0.4
weight2 = -0.7
weight3 = 1.2
```

But there are some settings that the researcher chooses before training.

Those are called **hyperparameters**.

For example:

```text
learning rate = 0.001
regularization strength = 100
number of trees = 500
```

Think of them as **settings on the machine**.

---

# 19. Why can't they use I-SPY1 to choose those settings?

Suppose they tried:

```text
CARF budget = 0.1
CARF budget = 0.5
CARF budget = 1.0
CARF budget = 2.0
```

and looked at I-SPY1 to decide:

> "0.5 gives the best result."

Then I-SPY1 is no longer a genuine independent test.

They have indirectly used the test data to design the model.

So they must do:

```text
I-SPY2
   ↓
choose settings
   ↓
train model
   ↓
FREEZE MODEL
   ↓
I-SPY1
   ↓
ONLY evaluate
```

This is the fundamental reason for keeping I-SPY1 untouched.

---

# 20. The whole protocol in one picture

This is probably the most useful way to remember it:

```text
                         I-SPY2
                    Development dataset
                           │
                           ▼
              ┌────────────────────────┐
              │ 5-fold cross-validation│
              │ repeated 5 times       │
              └───────────┬────────────┘
                          │
                 Outer training set
                          │
                          ▼
                 ┌─────────────────┐
                 │ Inner CV         │
                 │                 │
                 │ Clinical expert │
                 │ RNA expert      │
                 │ MRI expert      │
                 └────────┬────────┘
                          │
                          ▼
                Cross-fitted predictions
                          │
                          ▼
                 ┌─────────────────┐
                 │ Fusion models   │
                 │                 │
                 │ CARF            │
                 │ Stacking        │
                 │ etc.            │
                 └────────┬────────┘
                          │
                          ▼
                  Evaluate on outer
                     test patients
                          
                          
                    AFTER ALL OF THIS
                           │
                           ▼
                  Train final models
                    using ALL I-SPY2
                           │
                           ▼
                     ┌───────────┐
                     │  I-SPY1   │
                     │           │
                     │ No labels │
                     │ used for  │
                     │ training  │
                     │ or tuning │
                     └───────────┘
                           │
                           ▼
                    Final external
                      evaluation
```

---

## 21. The 8 ML terms you need from this paragraph

| Term                    | Very simple meaning                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **Training**            | Teaching the model using known examples                                              |
| **Testing**             | Checking the model on examples it didn't train on                                    |
| **Fold**                | One group of patients                                                                |
| **Cross-validation**    | Repeatedly training/testing using different groups                                   |
| **Stratified**          | Keep important proportions similar in each group                                     |
| **Expert**              | A model specialized in one modality                                                  |
| **Fusion**              | Combining predictions from multiple experts                                          |
| **Cross-fitting**       | Making predictions for each patient using a model that did not train on that patient |
| **Hyperparameter**      | A model setting chosen by the researcher                                             |
| **External evaluation** | Testing on a completely separate dataset                                             |

### The one distinction I especially want you to understand

There are **two levels** here:

```text
OUTER CV
"What happens to a completely unseen patient?"
                    ↓
               FINAL TEST
```

and inside the training portion:

```text
INNER CV
"How do I create honest predictions
to train the fusion model?"
```

That's why they use **nested cross-validation**.

If you understand that distinction, the protocol paragraph becomes much easier to read.
