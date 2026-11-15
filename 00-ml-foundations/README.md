# Phase 0 · ML Foundations

**Goal:** understand the few machine-learning ideas that everything else in this roadmap stands on: what a model is, how it is judged, how it learns (loss, gradient descent, backpropagation) and how it goes wrong (overfitting), so that the LLM and agent phases do not feel like magic.

**Who it's for:** complete beginners to machine learning, and anyone who has used AI tools but never looked at how a model is trained. This phase is optional if you already know these ideas; the quiz at the bottom is a quick way to check.

**Prerequisites:**
- Basic Python (lists, loops, functions).
- High-school math: averages, the idea of a slope, powers and the number `e`.
- Nothing else. No ML background needed.

**Time:** about 5-6 hours, including the code.

## Big idea
A machine-learning model is a function with adjustable numbers (weights). A **loss** measures how wrong its predictions are on training examples. **Gradient descent** repeatedly nudges the weights downhill on the loss, and **backpropagation** is the fast way to find which way is downhill for every weight at once. Train too hard on too little data and the model memorises instead of learning (**overfitting**), so we always judge it on data it has not seen. A large language model is this same loop at huge scale.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | AI, Machine Learning and Deep Learning | Beginner | [01-ai-ml-deep-learning.md](topics/01-ai-ml-deep-learning.md) |
| 02 | Loss and Gradient Descent | Beginner | [02-loss-and-gradient-descent.md](topics/02-loss-and-gradient-descent.md) |
| 03 | Neurons and Forward Propagation | Beginner | [03-neurons-and-forward-propagation.md](topics/03-neurons-and-forward-propagation.md) |
| 04 | Backpropagation | Intermediate | [04-backpropagation.md](topics/04-backpropagation.md) |
| 05 | Overfitting and Regularization | Intermediate | [05-overfitting-and-regularization.md](topics/05-overfitting-and-regularization.md) |

## Learning order
1. **What ML is (01).** The vocabulary, the three kinds of learning, and the train/validation/test split.
2. **How a model learns (02).** Loss and gradient descent, with one parameter you can follow by hand.
3. **What a network is (03).** Neurons, activations and the forward pass through a tiny 2-2-1 network.
4. **How a network learns (04).** Backpropagation on that same network, checked against finite differences.
5. **How learning goes wrong (05).** Overfitting, underfitting and the standard fixes.

Read them in order: 03 and 04 use the same network and the same numbers.

## Run the code
The demos in `code/` are offline, standard-library Python: no API key, no downloads, no randomness beyond fixed seeds. Snippets in topic pages that use `torch` or `scikit-learn` need those packages and are marked with `pip install`.

```bash
# from the repo root
python 00-ml-foundations/code/ml_gradient_descent.py   # 1-D descent, learning rates, linear regression
python 00-ml-foundations/code/ml_tiny_network.py       # 2-2-1 sigmoid network: forward, backprop, XOR
python 00-ml-foundations/code/ml_overfitting.py        # polynomial degrees, validation curve, ridge

python -m pytest -q 00-ml-foundations/code             # run the tests (needs: pip install pytest)
```

| File | What it shows | Used in topics |
|---|---|---|
| [`ml_gradient_descent.py`](code/ml_gradient_descent.py) | descent on (w-3)^2, divergence at a large learning rate, fitting a line | 02 |
| [`ml_tiny_network.py`](code/ml_tiny_network.py) | forward pass, hand-checkable backward pass, finite-difference gradient check, training on XOR | 03, 04 |
| [`ml_overfitting.py`](code/ml_overfitting.py) | normal equations, train vs validation error per degree, L2 (ridge) | 01, 05 |

## Try it
1. In `ml_gradient_descent.py`, try learning rates 0.3, 0.9 and 1.0 on the 1-D problem. For which `lr` does the loss stop improving?
2. Change the 1-D target from 3 to -2. Do the same learning rates still behave the same way?
3. In `ml_tiny_network.py`, change `LR` to 0.5 and then 2.0. How does the number of epochs needed to solve XOR change?
4. Change one starting weight in `INIT` and re-run the finite-difference test. Does backprop still match?
5. In `ml_overfitting.py`, use 30 training points instead of 10. Does degree 9 still overfit?
6. Try `lam` values between 1e-6 and 1e-2 on degree 9. Where is validation error lowest?

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. How are AI, machine learning and deep learning related?**
<details><summary>Answer</summary>
They are nested. AI is any technique that makes machines act intelligently; machine learning is the part that learns from data instead of hand-written rules; deep learning is the part of ML that uses neural networks with many layers.
</details>

**2. What separates supervised, unsupervised and reinforcement learning?**
<details><summary>Answer</summary>
Supervised learning has labelled examples and learns to predict the label. Unsupervised learning has no labels and finds structure such as clusters. Reinforcement learning has an agent that acts and learns from rewards.
</details>

**3. Why keep a test set that you look at only once?**
<details><summary>Answer</summary>
Choosing models and settings against a dataset makes its score optimistic. A test set used once, after all choices are made, gives an honest estimate for new data.
</details>

**4. What is a loss function? Name one for regression and one for classification.**
<details><summary>Answer</summary>
A number that is lower when predictions are better. Mean squared error is common for regression; cross-entropy (minus the log of the probability given to the correct class) for classification.
</details>

**5. Write the gradient descent update rule and say what the learning rate does.**
<details><summary>Answer</summary>
`w = w - lr * (dLoss/dw)`. The learning rate sets the step size: too small is slow, too large overshoots and can diverge.
</details>

**6. Minimising (w - 3)^2 from w = 0 with learning rate 0.1: what is w after one step?**
<details><summary>Answer</summary>
The gradient is 2(w - 3) = -6, so w = 0 - 0.1 * (-6) = 0.6.
</details>

**7. With learning rate 1.1 on the same problem, what happens and why?**
<details><summary>Answer</summary>
It diverges. Each step overshoots the minimum by more than the previous error (w goes 0, 6.6, -1.32, 8.18, ...), so the loss grows (9, 12.96, 18.66, ...). For this loss any learning rate above 1.0 diverges.
</details>

**8. What does one neuron compute?**
<details><summary>Answer</summary>
A weighted sum of its inputs plus a bias, passed through an activation function: `a = f(w · x + b)`.
</details>

**9. Why do networks need non-linear activations?**
<details><summary>Answer</summary>
Without them a stack of layers is equivalent to a single linear map, so depth adds nothing. Non-linearities let the network represent curved, complex functions such as XOR.
</details>

**10. What is the sigmoid's derivative in terms of its output, and why does that matter for deep sigmoid networks?**
<details><summary>Answer</summary>
`a * (1 - a)`, which is at most 0.25. Backprop multiplies one such factor per layer, so gradients can shrink toward zero in deep networks (vanishing gradients). ReLU, with slope 1 for positive inputs, helps.
</details>

**11. What is backpropagation?**
<details><summary>Answer</summary>
The chain rule applied from the loss backward through the network. It reuses intermediate results to give the gradient of the loss with respect to every weight at roughly the cost of one forward pass.
</details>

**12. In the 2-2-1 example, why did the weights on the first input get zero gradient for the input (0, 1)?**
<details><summary>Answer</summary>
A weight's gradient is its neuron's delta times the input flowing through it. That input was 0, so the product is 0: the weight had no influence on the loss for that example.
</details>

**13. How do you check that a hand-written backward pass is right?**
<details><summary>Answer</summary>
Gradient checking: compare each analytic gradient with the finite difference (L(w + h) - L(w - h)) / 2h for a small h such as 1e-5. They should match to many decimals.
</details>

**14. How do you recognise overfitting, and what are two fixes?**
<details><summary>Answer</summary>
Training error is low but validation error is much higher (and may rise as training continues). Fixes: more data, a simpler model, L2 regularization, early stopping or dropout.
</details>

**15. What does an L2 penalty do to the weights, and what happens if lambda is too large?**
<details><summary>Answer</summary>
It adds lambda times the sum of squared weights to the loss, shrinking the weights toward zero and smoothing the model. If lambda is too large the model underfits, because the weights are forced too close to zero.
</details>

More practice: [interview prep · LLM fundamentals](../08-interview-prep/01-llm-fundamentals.md).

**Prev:** [Start here](../00-start-here/README.md) · **Next:** [Phase 1 · RAG Foundations](../01-rag-foundations/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
