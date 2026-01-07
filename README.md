# SSMphony

## 📌 1. Overview

**SSMphony** is a Hindi-language text-to-speech (TTS) model built using a **Structured State Space (S4)** core for sequence modeling. It maps text input to speech waveforms via learned latent representations. It is designed for:

* High-quality TTS in Hindi
* Efficient long-range dependency modeling using S4 layers
* Scalable training and inference

The architecture consists of:

```
Text → Tokenizer → Embeddings → S4 Encoder/Decoder → Projection → Vocoder
```

The core innovation is the **S4 (Structured State Space)** layer which replaces traditional RNN/Transformer blocks for long sequence modeling.

---

## 📌 2. What Is Structured State Space (S4)?

S4 is a **state-space model** adapted for deep learning that can process very long sequences with linear-time complexity while retaining long-range signal information. It’s inspired by continuous-time dynamical systems.

### 🔹 Continuous Time State Space (CTSSM)

At the continuous level:

[
\dot{x}(t) = A x(t) + B u(t)
]
[
y(t) = C x(t) + D u(t)
]

Where:

* (u(t)) is input signal
* (x(t)) is latent state
* (y(t)) is output
* A, B, C, D are learned matrices

This describes **how hidden state evolves given input over time**.

### 🔹 Discrete Sequence Form

Discretizing with step (k):

[
x_k = \bar{A} x_{k-1} + \bar{B} u_k
]
[
y_k = \bar{C} x_k + \bar{D} u_k
]

Here:

* (\bar{A}, \bar{B}, \bar{C}, \bar{D}) are discrete state matrices
* Each new output depends on the current input and state

### 🔹 Efficient Computation with HiPPO & Diagonalization

S4 uses HiPPO matrices + diagonalization to *stabilize learning and cover long contexts*. Standard RNNs have vanishing/exploding gradients — S4’s math avoids that by modeling via **orthogonal/structured matrices**.

A key computational trick is:

[
\bar{A} = V \Lambda V^{-1}
]

Where (\Lambda) is diagonal — this **reduces complexity** and enables fast sequence convolution via FFT.

### 🔹 Convolutional View

S4 can be shown to implement:

[
y = k * u
]

Where convolution kernel (k) arises from state propagation. This means S4 acts like a **learned convolution filter** with super-long receptive field.

---

## 📌 3. SSMphony Architecture

Below is a typical sequence pipeline your repo likely includes based on common TTS structure and S4 modules:

```
input_text
   ↓ tokenizer
phoneme_ids / tokens
   ↓ embeddings (E)
embedded sequence
   ↓ S4 layers
latent features
   ↓ linear layers (projection)
mel-spectrogram
   ↓ vocoder
waveform
```

### 🎙️ 3.1 Text → Tokens

Given text string:

```
"नमस्ते दुनिया"
```

We map to a sequence of token IDs:

[
T = [t_1, t_2, ... , t_N]
]

These feed into an embedding layer:

[
E = W_e T
]

Where (W_e ∈ ℝ^{d×V}) and (V) is token vocabulary.

### 🎙️ 3.2 S4 Encoder/Decoder

Tokens → hidden:

[
H^{(0)} = E
]

Then for every layer (l):

[
H^{(l)} = \text{S4Layer}(H^{(l-1)})
]

Each S4Layer implements:

[
H^{(l)} = \text{Convolution}(H^{(l-1)}, k)
]

Where kernel (k) is derived from state-space propagation matrices with exponential stability.

### 🎙️ 3.3 Linear Projection → Mel Spectrogram

For time indices:

[
M = W_o H^{(L)} + b
]

Where:

* (M ∈ ℝ^{T×F}) is mel spectrogram
* (W_o, b) are trained output projection

Mel spectrogram represents frequency components over time.

### 🎙️ 3.4 Vocoder

The mel spectrogram is then converted to waveform using a neural vocoder (e.g., HiFi-GAN, WaveRNN). If yours is custom, it reconstructs audio samples.

---

## 📌 4. Training Objective

The model minimizes **spectrogram reconstruction loss**:

### 🟨 MSE Loss

[
\mathcal{L}*{MSE} = \frac{1}{TF} \sum*{t=1}^T \sum_{f=1}^F (M_{t,f} - \hat{M}_{t,f})^2
]


## 📌 5. Model Architecture (Diagram View)

```
┌──────────────────────┐
│        Text          │
│  (Hindi Sentence)    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     Tokenizer /      │
│   Phoneme Encoder    │
│  (Text → Tokens)     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    Embedding Layer   │
│  Tokens → Vectors    │
│  (N × d_model)       │
└──────────┬───────────┘
           ↓
╔══════════════════════╗
║      S4 ENCODER      ║
║ ─────────────────── ║
║  [ S4 Block × L ]    ║
║  • Long-range text   ║
║    dependency model  ║
║  • Residual + Norm   ║
╚═══════════╤══════════╝
            ↓
┌──────────────────────┐
│ Latent Representation│
│   (Linguistic Info)  │
│     N × d_model      │
└──────────┬───────────┘
           ↓
╔══════════════════════╗
║      S4 DECODER      ║
║ ─────────────────── ║
║  [ S4 Block × L ]    ║
║  • Duration modeling ║
║  • Prosody & rhythm  ║
║  • Temporal expand   ║
╚═══════════╤══════════╝
            ↓
┌──────────────────────┐
│   Linear Projection  │
│  d_model → Mel bins  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Mel-Spectrogram    │
│    (T × F bins)      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│       Vocoder        │
│ (HiFi-GAN / WaveRNN) │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Audio Waveform     │
│   (Speech Output)    │
└──────────────────────┘
```

---

### 🔹 Architectural Highlights

* **Encoder**
  Learns long-range linguistic structure from text using stacked S4 blocks.

* **Decoder**
  Expands text representations into time-aligned acoustic features (duration, prosody).

* **S4 Blocks**
  Replace attention and recurrence with structured state-space sequence modeling.

* **Parallel & Efficient**
  No autoregressive bottlenecks; scalable to long utterances.

---



## 📌 6. File Annotations

| Filename        | Purpose                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| `dataset.py`    | Loads text and speech data; text tokenization and mel extraction              |
| `s4.py`         | Core S4 layer implementation — includes state matrices, convolution utilities |
| `tts_model.py`  | Builds TTS model: embedding → S4 blocks → projection                          |
| `train.py`      | Training loop: loss, optimizer, batching                                      |
| `test_s4.py`    | to test the S4 layers                                                         |
| `test_model.py` | Unit tests for model sanity                                                   |

---

## 📌 7. Key Hyperparameters

| Parameter    | Meaning                      |
| ------------ | ---------------------------- |
| `d_model`    | Hidden dim (S4 feature size) |
| `L`          | Number of S4 blocks          |
| `lr`         | Learning rate                |
| `batch_size` | Samples per mini-batch       |
| `warmup`     | Warmup steps for optimizer   |
| `max_seq`    | Max text length              |





