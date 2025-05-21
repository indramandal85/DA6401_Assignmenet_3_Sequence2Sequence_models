# Attention-based Seq2Seq Model for Transliteration

This README covers the implementation and analysis of an attention-based sequence-to-sequence model for transliteration tasks (Questions 5-6).

## Attention Mechanism Implementation

The model implements a modular attention mechanism, supporting several classic attention strategies:
- **Luong Dot**: Computes attention scores as a simple dot product between encoder outputs and decoder hidden state
- **Luong General**: Introduces a learned linear transformation of encoder outputs before the dot product
- **Bahdanau (Concat) Attention**: Concatenates encoder outputs and decoder hidden state, transforms them with a feedforward layer

## Hyperparameter Tuning

Hyperparameters explored for the attention-based model:
- Embedding Size
- Hidden Layer Size
- Dropout Rate
- Learning Rate
- Bidirectionality
- Teacher Forcing Ratio
- Attention Method (Luong_general, Luong_dot, Bahdanau_concat)

### Best Configuration
- Embedding Size: 32  
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
- Attention Method: Luong_general  

## Test Results
- **Test Sequence Accuracy**: 43.61%  
- **Test Character Accuracy**: 76.35%  

## Performance Comparison with Vanilla Model
The attention-based model shows significant improvement over the vanilla seq2seq model:
- Vanilla Model: 37.61% sequence accuracy  
- Attention Model: 43.61% sequence accuracy  

## Error Corrections and Inferences

### Correct Word Structure Recognition
- `o k s o rgu l i` → `অক্ষরগুলি`
- `o t t a b a s h y a k` → `অত্যাবশ্যক`
- `o to r k i to` → `অতর্কিত`

### Enhanced Contextual Understanding
- `o n u c h h e d e` → `অনুচ্ছেদে`
- `o l in do` → `অলিন্দ`

### Specific Error Patterns Corrected
- Character omissions: `a b d u r` → `আব্দুর`
- Substitutions: `o ly mp i c s` → `অলিম্পিক্স`
- Word endings: `k o m e` → `কমে`

### Remaining Challenges
1. Rare or complex conjuncts  
2. Ambiguous transliterations  
3. Very long sequences  

## Attention Visualization

### Attention Heatmap Analysis
- **Diagonal Dominance**: Shows strong alignment between source and target tokens
- **Context Window Expansion**: Broader attention for complex characters
- **Start/End Token Focus**: High attention on sequence boundaries
- **Attention Shifting**: Dynamic context switching visible

### Connectivity Visualization
- Primary attention aligns with input token positions
- Secondary connections help with context awareness

## Predictions
All predictions are saved to:
```
predictions_attention/test_predictions_ATTENTION.csv
```
