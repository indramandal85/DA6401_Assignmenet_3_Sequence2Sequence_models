# Vanilla Seq2Seq Model for Transliteration

This README covers the implementation and analysis of a vanilla sequence-to-sequence model without attention for transliteration tasks (Questions 1-4).

## Model Architecture

The implemented RNN-based seq2seq model contains:
- Input layer for character embeddings
- Encoder RNN to sequentially encode the input character sequence (Latin)
- Decoder RNN that takes the last state of the encoder as input and produces output characters (Devanagari/Bangla)

The code is flexible, allowing customization of:
- Input character embedding dimensions
- Hidden state sizes for encoders and decoders
- Cell types (RNN, LSTM, GRU)
- Number of layers in encoder and decoder

## Computational Analysis

### Total Number of Computations
For a model with:
- Input embedding size: m  
- Hidden state size: k  
- Sequence length: T (same for input and output)  
- Vocabulary size: V  
- Encoder and decoder: 1 layer each  

The total number of computations per input-output sequence is:
```
2T(km + k²) + T(Vk)
```

### Total Number of Parameters
The total number of parameters in the network:
```
2Vm + 2km + 2k² + 2k + Vk + V
```

## Hyperparameter Tuning

Hyperparameters explored using Bayesian optimization:
- Embedding Size: 16, 32, 64, 256
- Hidden Size: 32, 64, 256, 512
- Layer Type: RNN, LSTM, GRU
- Encoder Layers: 1, 2, 3
- Decoder Layers: 1, 2, 3
- Dropout: 0.2, 0.3
- Learning Rate: 1e-4, 5e-4, 1e-3
- Teacher Forcing Ratio: 0.3, 0.5, 0.7, 1.0
- Bidirectional: True, False
- Beam Width: 1, 3, 5
- Validation Beam Search: True, False

### Best Configuration
- Embedding Size: 256  
- Hidden Size: 512  
- Layer Type: GRU  
- Encoder Layers: 3  
- Decoder Layers: 3  
- Dropout: 0.3  
- Learning Rate: 1e-4  
- Teacher Forcing Ratio: 0.7  
- Bidirectional: True  
- Beam Width: 3  
- Validation Beam Search: True  

## Key Observations

### Positive Observations
- LSTM and GRU layers outperform vanilla RNN
- Bidirectional encoders improve performance
- Larger embedding and hidden sizes yield better results
- Multi-layer architectures are beneficial
- Moderate dropout rates (0.2) are favorable
- Moderate teacher forcing ratios (0.5–0.7) work well
- Beam search improves sequence accuracy

### Negative Observations
- Significant variability in model performance
- Many hyperparameter combinations lead to poor performance
- Excessive dropout (0.3) may hinder performance
- Small hidden layer sizes underperform
- Single-layer models underperform

## Test Results
- **Test Sequence Accuracy**: 37.61%  
- **Test Character Accuracy**: 69.32%  
- **Test Loss**: 1.6305  

## Error Analysis

### Common Issues
- **Consonant vs Vowel Errors**: Struggles with consonant clusters
- **Sequence Length Issues**: Long sequences perform worse
- **Diacritical Mark Handling**: Improper placements
- **Transliteration Consistency**: Struggles with same word variations
- **Context Sensitivity**: Fails to leverage broader context
- **Rare Character Combinations**: Poor generalization

## Predictions
All predictions are saved to:
```
predictions_vanilla/test_predictions_without_attention.csv
```
