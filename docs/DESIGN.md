# Mamba-DeltaNet Hybrid Architecture Design

## Chosen Arrangement: Interleaved (Alternating)

Each layer alternates between a Mamba block and a DeltaNet block.
Layer 1 is Mamba, Layer 2 is DeltaNet, Layer 3 is Mamba, Layer 4 is DeltaNet, and so on.
Both the encoder and decoder stacks follow this interleaved pattern.
In the decoder, each block is followed by a cross-attention sublayer that attends to the encoder output.

## Hypothesis

Cipher decryption decomposes into two complementary sub-problems.

The first is local sequential pattern modelling, which is best handled by Mamba.
The cipher applies consistent character-level transformations, and recovering the original text requires tracking how local context influences each substituted symbol.
Mamba's selective state space model uses a 1D causal convolution and input-dependent gating parameters (delta, B, C) to capture smooth local context and propagate it through its recurrent hidden state.
The selectivity mechanism allows the model to decide how much of the current token to let into the state and how much of the past state to retain, which is well-suited to tracking positional and contextual drift in the cipher.

The second sub-problem is associative symbol binding, which is best handled by DeltaNet.
Fundamentally, decryption is a lookup: each cipher token maps to a plaintext token according to a fixed (but unknown) substitution function.
DeltaNet's delta rule maintains a matrix memory S that behaves as an explicit key-value associative store.
At each step it erases the previous value associated with the current key and writes the new value, enabling the model to learn and retrieve precise cipher-to-plaintext bindings in-context.
This write-erase mechanism is structurally better suited to symbol-level memorisation than the additive updates of standard linear attention.

## Why Interleaving?

A purely sequential arrangement, in which all Mamba layers come first followed by all DeltaNet layers, would force each sub-network to process its portion of the computation on its own input representations without benefit of the other mechanism.
Interleaving ensures that at every layer depth, the sequential context built by Mamba feeds directly into DeltaNet's associative binding step, and conversely the retrieved symbol bindings from DeltaNet inform the next Mamba layer's state update.
This tight coupling at every depth encourages the two mechanisms to co-adapt and jointly optimise their complementary roles rather than learning independent representations.

## Expected Behaviour

We expect the hybrid to outperform pure Mamba on associative recall tasks, specifically on binding specific cipher symbols to their plaintext counterparts, because DeltaNet layers provide explicit write-erase memory.
We expect the hybrid to outperform pure DeltaNet on longer sequences where smooth state propagation across many positions matters, because Mamba layers maintain a compressed but continuous context.
We expect the interleaved hybrid to approach Transformer accuracy with better asymptotic scaling (linear vs quadratic complexity) on long sequences, and with a parameter count that remains comparable to the other single-architecture baselines.

