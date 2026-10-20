# ADR 0003: Word-level vocabulary for the small transformer LM

**Status:** accepted · **PRD:** 14.1, 14.3 (deviation)

The PRD suggests byte-pair encoding for the L2 transformer. GazeOS uses a
30k-word vocabulary instead:

- The keyboard needs probabilities of whole next words. With BPE, scoring a
  candidate word needs several forward passes (one per sub-word), which breaks the
  30 ms budget once the decoder scores dozens of candidates.
- The prediction stack already handles out-of-vocabulary words through the
  trie (L0), the user dictionary and the noisy-channel decoder.

Revisit for Hindi and Gujarati (FR-31), where a word-level vocabulary is a
poor fit; a sub-word model with cached prefix scores would be needed there.
