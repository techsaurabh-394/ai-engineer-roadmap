"""Animation content for phase 0 (ML foundations)."""
from anim_core import P, L, B

# ---------------- Phase 0: ML foundations ----------------
P("01-ai-ml-deep-learning", "AI, ML and deep learning", "Nested ideas, and why we split the data three ways.", """
AI|the big goal|1. AI means machines doing things we call smart|Hand-written rules count too.
ML|learn from data|2. ML learns the rules from examples|You show labelled emails instead of writing rules.
Deep|many layers|3. Deep learning is ML with many-layered networks|It powers image models and LLMs.
Split|train/val/test|4. Split the data: train, validation and test|For example 70%, 15% and 15%.
Judge|unseen data|5. Judge the model only on data it has not seen|Test set: look once, at the very end.
""", color="teal")

L("02-loss-and-gradient-descent", "Loss and gradient descent", "Walk downhill on the loss, one small step at a time.", """
Predict|1. The model makes a prediction|With its current weights.
Loss|2. Loss turns the error into one number|MSE for numbers, cross-entropy for classes.
Gradient|3. The gradient gives the slope for each weight|It points toward higher loss.
Step|4. Move each weight against the slope: w = w - lr * g|Too big a step overshoots and diverges.
Repeat|5. Repeat until the loss stops falling|Example: w goes 0, 0.6, 1.08 toward 3.
""", color="amber")

P("03-neurons-and-forward-propagation", "Neurons and forward propagation", "Weighted sum, bias, activation, layer after layer.", """
Inputs|x1, x2|1. The inputs enter the network|Example: x = (0, 1).
Weights|w * x + b|2. Each neuron computes a weighted sum plus a bias|Hidden neuron 1: z = -0.3.
Activation|sigmoid|3. An activation squashes the sum|sigmoid(-0.3) = 0.4256.
Layer|2 hidden|4. Both hidden outputs feed the output neuron|Hidden outputs: 0.4256 and 0.6457.
Output|prediction|5. The output neuron gives the prediction|0.4901 before any training.
""", color="blue")

L("04-backpropagation", "Backpropagation", "The chain rule, run backward from the loss.", """
Forward|1. Run forward and save every z and a|Prediction 0.4901, loss 0.1300.
Output delta|2. Output delta: d2 = (a2 - y) * a2 * (1 - a2)|Here d2 = -0.1274.
Hidden deltas|3. Push the delta back through the weights|Deltas: -0.0218 and +0.0175.
Gradients|4. Each weight's gradient is delta times its input|Reuse of results keeps it fast.
Update|5. Update every weight: w = w - lr * gradient|Loss on this example falls to 0.1166.
""", color="pink")

B("05-overfitting-and-regularization", "Overfitting and regularization", "Validation error is U-shaped; a small L2 penalty tames degree 9.",
  ["deg 1", "deg 3", "deg 5", "deg 8", "deg 9 + L2"],
  [("Underfit", [0.45, 0.16, 0.16, 0.21, 0.16], [1, 0, 0, 0, 0],
    "1. A straight line underfits: validation error 0.45", "It is too simple to follow a sine wave."),
   ("Sweet spot", [0.45, 0.16, 0.16, 0.21, 0.16], [0, 1, 1, 0, 0],
    "2. Degrees 3 to 5 do best: validation error about 0.16", "Flexible enough, but not too flexible."),
   ("Overfit", [0.45, 0.16, 0.16, 0.21, 0.16], [0, 0, 0, 1, 0],
    "3. Degree 8 climbs to 0.21; degree 9 reaches 3.29 (off this chart)", "It memorises the 10 training points and the noise."),
   ("Regularized", [0.45, 0.16, 0.16, 0.21, 0.16], [0, 0, 0, 0, 1],
    "4. Degree 9 with a small L2 penalty: validation error 0.16", "Training error rises a little; new-data error drops a lot.")],
  color="teal")
