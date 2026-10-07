# 03 - Deep learning (course Modules 13-16)

Hands-on: [Lab 03 - MLP on circles](../../part1-ai-foundations/lab03_mlp_circles_decision_boundary.py).

## Foundations
- DL = subset of ML that trains **Artificial Neural Networks (ANN)**. ANNs process **raw data** and extract features automatically; in classic ML you specify features by hand.
- DL splits data into batches processed in parallel -> scales to huge data.

**History:** 1950s perceptron - 1980s backpropagation - 1990s CNNs - 2000s GPUs - 2010s cheap GPUs fuel adoption - 2012 AlexNet / Deep Q-Network - 2016+ generative use cases - today LLMs.

| Data | Applications | Best architecture |
|---|---|---|
| Images | classification, detection, segmentation, face recognition | **CNN** |
| Text (general) | translation, sentiment | Transformer / LSTM / RNN |
| Generative text | summarisation, Q&A | **Transformer** |
| Image generation | text-to-image | Transformer, GAN, diffusion |
| Audio | speech-to-text, music generation | RNN/LSTM/Transformer |

## ANN building blocks
Input layer (required) - hidden layers (optional, many) - output layer (required) - **neurons** - **weights** - **bias** - **activation functions** (turn the weighted sum into the neuron's output; ReLU adds non-linearity).
Handwritten digits: 28x28 = 784 inputs -> two hidden layers of 16 -> 10 outputs. **Backpropagation**: wrong neuron fires -> compute error -> adjust weights -> repeat over thousands of images = training.

## Architecture cheat sheet
| Architecture | Purpose |
|---|---|
| FNN / MLP | simplest network |
| **CNN** | local patterns in images/video |
| **RNN** | sequences; feedback loop keeps hidden state |
| **LSTM** | RNN variant for long-term dependencies |
| Autoencoder | unsupervised features, compression, anomaly detection |
| **GAN** | generate realistic synthetic data |
| **Transformer** | state of the art for NLP |

## CNN
Works natively on 2-D grids (an ANN flattens images to 1-D). Layers: input -> [convolution + ReLU -> pooling] x N -> fully connected -> softmax.
Robot-house-inspector analogy: **convolution** (blueprint detector, filters find edges/corners) - **activation** (pattern highlighter) - **pooling** (room summariser) - **fully connected** (house expert) - **softmax** (guess maker: probabilities) - **dropout** (quality checker, reduces overfitting).
Limitations: expensive to train, overfits on small/imbalanced data, "black box", sensitive to small input changes. Uses: classification, detection, segmentation, face recognition, medical imaging, self-driving, satellite imagery.

## Demo: MLP on `make_circles` (Module 15)
Two concentric rings cannot be separated by a line. 1 hidden neuron -> predicts almost everything as class 0; 2-3 neurons -> partial; 4+ -> accurate non-linear boundary. Parameters: `n_samples=300`, `noise`, `factor` (gap between circles), `random_state`; ReLU activation; boundary drawn with `contourf` on a 100x100 grid. **Reproduced in Lab 03.**

## Sequence models (Module 16)
Sequence data = ordered events (text, audio, time series). RNN types: one-to-many (music generation), many-to-one (sentiment), many-to-many (translation, NER). RNNs suffer **vanishing gradients** on long sequences -> **LSTM**: memory cell + **input gate** (what to add), **forget gate** (what to discard), **output gate** (what to expose).
