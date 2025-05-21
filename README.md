# Sequence to Sequence Model with & without Attention Mechanism Analysis

## Student Details
- **Name**: Indra Mandal
- **Roll No**: ED24S014

## Project Links
- [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/seq2seq_without_Attention_mechanism/reports/Assignment-3-Seq2seq-Model---VmlldzoxMjc1NDkwOQ)
- [Github Repository](https://github.com/indramandal85/DA6401_Assignmenet_3_Sequence2Sequence_models.git)

## Project Overview
This project implements and analyzes sequence-to-sequence models for transliteration tasks using the Dakshina dataset. The goal is to build a system that can convert romanized text (Latin script) to native script (Devanagari/Bangla).

### Objectives
1. Model sequence-to-sequence learning problems using Recurrent Neural Networks
2. Compare different RNN cell types (vanilla RNN, LSTM, GRU)
3. Understand how attention networks overcome limitations of vanilla seq2seq models
4. Visualize interactions between different components in RNN-based models

## Repository Structure
```
├── predictions_vanilla/
│   └── test_predictions_without_attention.csv
├── predictions_attention/
│   └── test_predictions_ATTENTION.csv
├── README.md
├── README_vanilla.md
├── README_attention.md
└── [code files]
```

## Models Implemented
1. **Vanilla Seq2Seq Model**: A basic encoder-decoder architecture without attention mechanism
2. **Attention-based Seq2Seq Model**: Enhanced model with various attention mechanisms (Luong, Bahdanau)

## Key Results
- **Vanilla Model Test Accuracy**: 37.61% (sequence-level), 69.32% (character-level)
- **Attention Model Test Accuracy**: 43.61% (sequence-level), 76.35% (character-level)

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Weights & Biases (wandb)

## Usage
Please refer to the specific README files for detailed instructions on each model:
- [Vanilla Seq2Seq Model](./README_vanilla.md)
- [Attention-based Seq2Seq Model](./README_attention.md)

