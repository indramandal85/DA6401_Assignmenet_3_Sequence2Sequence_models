import numpy as np
import pandas as pd
import torch
import random
import time
from tqdm import tqdm
from datetime import datetime
import copy
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb
wandb.login(key = '5df7feeffbc5b918c8947f5fe4bab4b67ebfbb69')


train_df =('/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_3/predictions_attention/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv')
dev_df = ('/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_3/predictions_attention/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv')
test_df = ('/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_3/predictions_attention/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv')



# Special tokens
SOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'

class SequenceDataPreprocessor:
    def __init__(self, path, input_vocab=None, output_vocab=None):
        self.path = path
        self.input_token_to_idx = input_vocab
        self.output_token_to_idx = output_vocab

    def read_data(self, file_path):
        df = pd.read_csv(file_path, sep="\t", names=["target", "input", "count"]).astype(str)
        inputs, outputs = [], []
        for _, row in df.iterrows():
            inp = list(row['input'])
            out = [SOS] + list(row['target']) + [EOS]
            inputs.append(inp)
            outputs.append(out)
        return inputs, outputs

    def build_vocab(self, sequences):
        all_tokens = [token for seq in sequences for token in seq]
        counts = Counter(all_tokens)

        specials_list = [PAD, SOS, EOS, UNK]
        for token in specials_list:
            counts[token] = counts.get(token, 1)

        normal_tokens = sorted([tok for tok in counts if tok not in specials_list])
        tokens = specials_list + normal_tokens

        return {token: idx for idx, token in enumerate(tokens)}

    def encode_sequences(self, sequences, vocab):
        unk_idx = vocab.get(UNK, vocab.get(PAD, 0))  # Fallback
        return [torch.tensor([vocab.get(token, unk_idx) for token in seq], dtype=torch.long) for seq in sequences]

    def pad_batch(self, batch, pad_idx):
        return pad_sequence(batch, batch_first=True, padding_value=pad_idx)

    def prepare_tensors(self):
        inputs, targets = self.read_data(self.path)

        # Build vocab if not provided
        if self.input_token_to_idx is None:
            self.input_token_to_idx = self.build_vocab(inputs)
        if self.output_token_to_idx is None:
            self.output_token_to_idx = self.build_vocab(targets)

        # Check PAD is in vocab
        if PAD not in self.input_token_to_idx or PAD not in self.output_token_to_idx:
            raise ValueError("PAD token not found in vocab. Ensure special tokens are added in build_vocab.")

        input_ids = self.encode_sequences(inputs, self.input_token_to_idx)
        target_ids = self.encode_sequences(targets, self.output_token_to_idx)

        input_tensor = self.pad_batch(input_ids, self.input_token_to_idx[PAD])
        target_tensor = self.pad_batch(target_ids, self.output_token_to_idx[PAD])

        return input_tensor, target_tensor, self.input_token_to_idx, self.output_token_to_idx


class Datasets(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.input_tensor.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.input_tensor[idx], self.target_tensor[idx]



class Encoder(nn.Module):
    def __init__(self, input_size, layer_type, emb_dim, hidden_layers_size, num_encod_layers, dropout_rate, pad_index, bidirectional=False):
        super().__init__()
        self.layer_type = layer_type
        self.layers = self.layer_mode(layer_type)
        self.bidirectional = bidirectional
        self.num_encod_layers = num_encod_layers
        self.hidden_size = hidden_layers_size

        self.embed = nn.Embedding(input_size, emb_dim, padding_idx=pad_index)
        self.layer = self.layers(
            emb_dim, 
            hidden_layers_size, 
            num_encod_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_encod_layers > 1 else 0
        )

    def layer_mode(self, layer_type):
        layer_type = layer_type.lower()
        if layer_type == "rnn":
            return nn.RNN
        elif layer_type == "lstm":
            return nn.LSTM
        else:
            return nn.GRU

    def forward(self, input_seq):
        embed = self.embed(input_seq)
        if self.layer_type == "lstm":
            outputs, (hidden, cell) = self.layer(embed)
        else:
            outputs, hidden = self.layer(embed)
            cell = None
        return outputs, hidden, cell
    


class Attention(nn.Module):
    def __init__(self, method, encoder_hidden_size, decoder_hidden_size):
        super().__init__()
        self.method = method
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        if self.method == 'Luong_general':
            self.Wa = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        elif self.method == 'Bahdanau_concat':
            self.Wa = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
            self.v = nn.Linear(decoder_hidden_size, 1, bias=False)
        elif self.method != 'Luong_dot':
            raise ValueError("Invalid attention method")
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch_size, decoder_hidden_size)
        # encoder_outputs: (batch_size, src_len, encoder_hidden_size)
        
        if self.method == 'Luong_dot':
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == 'Luong_general':
            transformed_encoder = self.Wa(encoder_outputs)
            scores = torch.bmm(transformed_encoder, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == 'Bahdanau_concat':
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            combined = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
            energy = torch.tanh(self.Wa(combined))
            scores = self.v(energy).squeeze(2)
        
        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, layer_type, emb_dim, hidden_layers_size,
                 num_decod_layers, dropout_rate, pad_index, encoder_hidden_size, 
                 bidirectional=False, attention_method='Luong_general'):
        super().__init__()
        self.layer_type = layer_type.lower()
        self.bidirectional = bidirectional
        self.num_layers = num_decod_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.attention_method = attention_method
        self.hidden_size = hidden_layers_size

        self.embed = nn.Embedding(input_size, emb_dim, padding_idx=pad_index)
        
        # Attention layer
        self.attention = Attention(
            method=attention_method,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=hidden_layers_size * (2 if bidirectional else 1)
        )
        
        # RNN input size: emb_dim + encoder_hidden_size (context)
        rnn_input_size = emb_dim + encoder_hidden_size
        
        rnn_cls = self.layer_mode(self.layer_type)
        self.layer = rnn_cls(
            rnn_input_size,
            hidden_layers_size,
            num_decod_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_decod_layers > 1 else 0
        )
        
        rnn_output_dim = hidden_layers_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_dim, output_size)

    def layer_mode(self, layer_type):
        if layer_type == "rnn":
            return nn.RNN
        elif layer_type == "lstm":
            return nn.LSTM
        else:
            return nn.GRU

    def forward(self, inputs: torch.LongTensor, hidden, encoder_outputs, cell=None):
        inputs = inputs.unsqueeze(1)  # (batch, 1)
        embed = self.embed(inputs)     # (batch, 1, emb_dim)
        
        # Prepare decoder hidden state for attention
        if self.bidirectional:
            # Reshape for multi-layer bidirectionality
            hidden_reshaped = hidden.view(
                self.num_layers, 2,  # (num_layers, num_directions)
                -1,                  # Batch size
                self.hidden_size
            )
            # Concatenate last layer's forward/backward states
            hidden_combined = torch.cat([
                hidden_reshaped[-1, 0, :, :],  # Forward direction
                hidden_reshaped[-1, 1, :, :]   # Backward direction
            ], dim=1)
        else:
            hidden_combined = hidden[-1]  # (batch_size, hidden_size)
        
        context, attn_weights = self.attention(hidden_combined, encoder_outputs)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([embed, context], dim=2)
        
        if self.layer_type == "lstm":
            outputs, (hidden, cell) = self.layer(rnn_input, (hidden, cell))
        else:
            outputs, hidden = self.layer(rnn_input, hidden)
            cell = None
        
        predict_word = self.fc(outputs.squeeze(1))
        return predict_word, hidden, cell
    



class Sequence2Sequence(nn.Module):
    def __init__(self, encoder, decoder, output_vocab):
        super().__init__()
        self.output_vocab = output_vocab
        self.encoder = encoder
        self.decoder = decoder
        self.layer_type = encoder.layer_type.lower()

    def adjust_hidden(self, hidden, desired_layers):
        current_layers = hidden.size(0)
        if current_layers < desired_layers:
            zeros = torch.zeros(desired_layers - current_layers, 
                              hidden.size(1), 
                              hidden.size(2),
                              device=hidden.device,
                              dtype=hidden.dtype)
            adjusted = torch.cat([hidden, zeros], dim=0)
        else:
            adjusted = hidden[:desired_layers]
        return adjusted

    def forward(self, input_sequence, target_sequence, teacher_force_ratio=0.5):
        batch_size = input_sequence.size(0)
        target_len = target_sequence.size(1)
        outputs = torch.zeros(batch_size, target_len, len(self.output_vocab)).to(input_sequence.device)
        
        # Encoder forward
        encoder_outputs, hidden, cell = self.encoder(input_sequence)
        
        # Handle bidirectional encoder
        # if self.encoder.bidirectional:
        #     encoder_outputs = encoder_outputs[:, :, :self.encoder.hidden_size] 
        
        # Adjust hidden states for decoder
        encoder_directions = 2 if self.encoder.bidirectional else 1
        decoder_directions = 2 if self.decoder.bidirectional else 1
        encoder_total = self.encoder.num_encod_layers * encoder_directions
        decoder_total = self.decoder.num_layers * decoder_directions
        
        hidden = self.adjust_hidden(hidden, decoder_total)
        cell = self.adjust_hidden(cell, decoder_total) if cell is not None else None
        
        x = target_sequence[:, 0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, encoder_outputs, cell)
            outputs[:, t] = output
            x = target_sequence[:, t] if random.random() < teacher_force_ratio else output.argmax(1)
        return outputs

    def beam_search_decode(self, input_sequence, sos_token, eos_token, beam_width=3, max_len=30):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            device = input_sequence.device
            encoder_outputs, hidden, cell = self.encoder(input_sequence)
            
            # Handle bidirectional encoder outputs
            # if self.encoder.bidirectional:
            #     encoder_outputs = encoder_outputs[:, :, :self.encoder.hidden_size] + encoder_outputs[:, :, self.encoder.hidden_size:]
            
            # Adjust hidden states
            encoder_directions = 2 if self.encoder.bidirectional else 1
            decoder_directions = 2 if self.decoder.bidirectional else 1
            encoder_total = self.encoder.num_encod_layers * encoder_directions
            decoder_total = self.decoder.num_layers * decoder_directions
            
            hidden = self.adjust_hidden(hidden, decoder_total)
            cell = self.adjust_hidden(cell, decoder_total) if cell is not None else None

            beams = [([sos_token], 0.0, hidden.repeat(1, beam_width, 1), cell.repeat(1, beam_width, 1) if cell is not None else None)]
            completed_sequences = []

            for _ in range(max_len):
                temp_beams = []
                for seq, score, h, c in beams:
                    if seq[-1] == eos_token:
                        completed_sequences.append((seq, score))
                        continue

                    last_token = torch.LongTensor([seq[-1]]).to(device)
                    out, h_new, c_new = self.decoder(last_token, h, encoder_outputs.repeat(beam_width, 1, 1), c)
                    log_probs = torch.log_softmax(out, dim=1)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        token = top_indices[0][i].item()
                        new_seq = seq + [token]
                        new_score = score + top_log_probs[0][i].item()
                        temp_beams.append((new_seq, new_score, h_new[:, i:i+1, :], c_new[:, i:i+1, :] if c_new is not None else None))

                beams = sorted(temp_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                if all(seq[-1] == eos_token for seq, _, _, _ in beams):
                    completed_sequences.extend(beams)
                    break

            if not completed_sequences:
                completed_sequences = beams

            best_sequence = max(completed_sequences, key=lambda x: x[1])[0]
            return best_sequence



class AccuracyCalculator:
    def __init__(self, eos_token: str, pad_token: str, vocab_out: dict, device: torch.device):
        """
        eos_token: the string for <eos>
        pad_token: the string for <pad>
        vocab_out: token->index mapping for your output vocab
        device:     torch.device (e.g. 'cuda' or 'cpu')
        """
        self.device = device
        self.eos_idx = vocab_out[eos_token]
        self.pad_idx = vocab_out[pad_token]

    def _trim_batch_at_eos(self, sequences: torch.LongTensor):
        """
        sequences: (batch_size, seq_len)
        Returns: list of 1D LongTensors, each trimmed to include its first <eos> (if any),
                 or the full length if no <eos> appears.
        """
        batch_size, seq_len = sequences.size()
        trimmed = []
        # move to CPU & numpy for easy indexing
        seqs = sequences.detach().cpu().tolist()
        for seq in seqs:
            if self.eos_idx in seq:
                end = seq.index(self.eos_idx) + 1
                trimmed.append(torch.tensor(seq[:end], dtype=torch.long, device=self.device))
            else:
                trimmed.append(torch.tensor(seq, dtype=torch.long, device=self.device))
        return trimmed

    def compute_accuracy(self,
                         predictions: torch.LongTensor,
                         targets:     torch.LongTensor
                         ) -> dict:
        """
        predictions: (batch_size, seq_len) of token-indices, already argmaxed
        targets:     (batch_size, seq_len) of token-indices, contains <sos>…<eos> and padding
        """
        predictions = predictions.to(self.device)
        targets     = targets.to(self.device)

        batch_size, seq_len = targets.shape

        # Ensure predictions and targets have the same length by trimming or padding predictions
        if predictions.size(1) > seq_len:
            predictions = predictions[:, :seq_len]
        elif predictions.size(1) < seq_len:
            pad_len = seq_len - predictions.size(1)
            padding = torch.full((predictions.size(0), pad_len), self.pad_idx, dtype=predictions.dtype, device=predictions.device)
            predictions = torch.cat([predictions, padding], dim=1)

        # 1) Character-level accuracy (ignoring PAD completely)
        nonpad_mask   = targets != self.pad_idx                # (B, L) bool
        char_correct  = ((predictions == targets) & nonpad_mask).sum().item()
        char_total    = nonpad_mask.sum().item()
        char_accuracy = char_correct / char_total if char_total > 0 else 0.0

        # 2) Sequence-level accuracy
        #    Trim both preds & targets at each target's <eos>, then compare exactly.
        pred_trimmed = self._trim_batch_at_eos(predictions)
        targ_trimmed = self._trim_batch_at_eos(targets)

        seq_correct = 0
        for p_seq, t_seq in zip(pred_trimmed, targ_trimmed):
            if p_seq.size(0) == t_seq.size(0) and torch.equal(p_seq, t_seq):
                seq_correct += 1

        seq_accuracy = seq_correct / batch_size if batch_size > 0 else 0.0

        return {
            'sequence_accuracy':   seq_accuracy,
            'character_accuracy':  char_accuracy,
            'correct_sequences':   seq_correct,
            'total_sequences':     batch_size,
            'correct_characters':  char_correct,
            'total_characters':    char_total
        }



class Train_Model:
    def __init__(self, seq2seq, dataloader, optimizer, loss_fn, acc_calculator, device):
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.acc_calculator = acc_calculator
        self.device = device
        self.seq2seq = seq2seq

    def count_params(self,model):
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    def train(self, teacher_force_ratio=0.5):
        self.seq2seq.to(self.device)
        self.seq2seq.train()

        epoch_loss = 0
        seq_acc = 0
        character_acc = 0
        total_seqs = 0
        total_chars = 0

        progress_bar = tqdm(self.dataloader, desc="Training Batches")

        for input_batch, target_batch in progress_bar:
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            self.optimizer.zero_grad()

            output = self.seq2seq(input_batch, target_batch, teacher_force_ratio)
            _, predicted = torch.max(output, dim=2)

            # Calculate accuracy ignoring SOS token
            predicted_trimmed = predicted[:, 1:]
            target_trimmed = target_batch[:, 1:]

            # Flatten for loss calculation
            output_flat = output.view(-1, output.shape[-1])
            target_flat = target_batch.reshape(-1).to(self.device)

            loss = self.loss_fn(output_flat, target_flat)
            acc = self.acc_calculator.compute_accuracy(predicted_trimmed, target_trimmed)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.seq2seq.parameters(), max_norm=1)
            self.optimizer.step()

            epoch_loss += loss.item()
            seq_acc += acc['correct_sequences']
            total_seqs += acc['total_sequences']
            character_acc += acc['correct_characters']
            total_chars += acc['total_characters']

            avg_seq_acc = seq_acc / total_seqs if total_seqs > 0 else 0.0
            avg_char_acc = character_acc / total_chars if total_chars > 0 else 0.0

            progress_bar.set_postfix({
                'Train_loss': loss.item(),
                'seq_acc': f"{avg_seq_acc:.2%}",
                'char_acc': f"{avg_char_acc:.2%}"
            })

        return (epoch_loss / len(self.dataloader), 
                avg_char_acc, 
                avg_seq_acc)
    



class Evaluate_Model:
    def __init__(self, seq2seq, dataloader, loss_fn, acc_calculator, device):
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.acc_calculator = acc_calculator
        self.device = device
        self.seq2seq = seq2seq

    def evaluate(self, beam_search=False, beam_width=3):
        self.seq2seq.eval()
        self.seq2seq.to(self.device)

        epoch_loss = 0
        seq_acc = 0
        character_acc = 0
        total_seqs = 0
        total_chars = 0

        progress_bar = tqdm(self.dataloader, desc="Evaluation Batches")

        with torch.no_grad():
            for input_batch, target_batch in progress_bar:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                if beam_search:
                    decoded_batch = []
                    for i in range(input_batch.size(0)):
                        predicted_ids = self.seq2seq.beam_search_decode(
                            input_batch[i].unsqueeze(0),
                            sos_token=self.seq2seq.output_vocab[SOS],
                            eos_token=self.seq2seq.output_vocab[EOS],
                            beam_width=beam_width
                        )
                        decoded_batch.append(torch.tensor(predicted_ids, device=self.device))

                    max_len = max(len(seq) for seq in decoded_batch)
                    predicted_tensor = torch.full((input_batch.size(0), max_len), 
                                      fill_value=self.seq2seq.output_vocab[PAD], 
                                      device=self.device)
                    for i, seq in enumerate(decoded_batch):
                        predicted_tensor[i, :len(seq)] = seq
                else:
                    output = self.seq2seq(input_batch, target_batch, teacher_force_ratio=0)
                    _, predicted = torch.max(output, dim=2)
                    predicted_tensor = predicted

                # Pad/cut predictions to match target length
                predicted_tensor = predicted_tensor[:, :target_batch.size(1)]
                if predicted_tensor.size(1) < target_batch.size(1):
                    pad = torch.full((predicted_tensor.size(0), 
                                    target_batch.size(1) - predicted_tensor.size(1)),
                                    self.seq2seq.output_vocab[PAD], 
                                    device=self.device)
                    predicted_tensor = torch.cat([predicted_tensor, pad], dim=1)

                # Calculate loss
                output = self.seq2seq(input_batch, target_batch, teacher_force_ratio=0)
                output_flat = output.view(-1, output.shape[-1])
                target_flat = target_batch.view(-1)
                loss = self.loss_fn(output_flat, target_flat)

                # Calculate accuracy
                pred_trimmed = predicted_tensor[:, 1:]
                target_trimmed = target_batch[:, 1:]
                acc = self.acc_calculator.compute_accuracy(pred_trimmed, target_trimmed)

                epoch_loss += loss.item()
                seq_acc += acc['correct_sequences']
                total_seqs += acc['total_sequences']
                character_acc += acc['correct_characters']
                total_chars += acc['total_characters']

                avg_seq_acc = seq_acc / total_seqs if total_seqs > 0 else 0.0
                avg_char_acc = character_acc / total_chars if total_chars > 0 else 0.0

                progress_bar.set_postfix({
                    'Val_loss': loss.item(),
                    'seq_acc': f"{avg_seq_acc:.2%}",
                    'char_acc': f"{avg_char_acc:.2%}"
                })

        return (epoch_loss / len(self.dataloader), 
                avg_char_acc, 
                avg_seq_acc)
    

class Build_Model:
    def __init__(self, 
                 sequence_data_preprocessor,
                 encoder_class, 
                 decoder_class, 
                 seq2seq_class, 
                 attention_class,
                 batch_size,
                 train_path,
                 val_path,
                 device="cpu"):
        
        self.sequence_data_preprocessor = sequence_data_preprocessor
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.seq2seq_class = seq2seq_class
        self.attention_class = attention_class
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.device = device

        # Process datasets
        train_processor = self.sequence_data_preprocessor(self.train_path)
        self.train_input_tensor, self.train_target_tensor, self.input_vocab, self.output_vocab = train_processor.prepare_tensors()

        val_processor = self.sequence_data_preprocessor(self.val_path, input_vocab=self.input_vocab, output_vocab=self.output_vocab)
        self.val_input_tensor, self.val_target_tensor, _, _ = val_processor.prepare_tensors()

        # Create datasets and dataloaders
        train_data = Datasets(self.train_input_tensor, self.train_target_tensor)
        val_data = Datasets(self.val_input_tensor, self.val_target_tensor)

        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

    def epoch_time(self, start_time, end_time):
        elapsed = end_time - start_time
        return int(elapsed // 60), int(elapsed % 60)

    def build(self,
              emb_size, layer_type, hidden_layers_size,
              num_encod_layers, num_decod_layers,
              dropout_rate, epochs, learning_rate,
              teacher_force_ratio=0, bidirectional=False,
              save_path='best_model.pt', patience=3,
              val_beam_search=False, beam_width=3,
              testing_phase=False, test_path=None,
              test_beam_search=False, wandb_log=False,
              attention_method='Luong_general'):
        
        # Instantiate Encoder
        encoder = self.encoder_class(
            input_size=len(self.input_vocab),
            layer_type=layer_type,
            emb_dim=emb_size,
            hidden_layers_size=hidden_layers_size,
            num_encod_layers=num_encod_layers,
            dropout_rate=dropout_rate,
            pad_index=self.input_vocab[PAD],
            bidirectional=bidirectional
        ).to(self.device)

        # Calculate encoder hidden size for attention
        encoder_hidden_size = hidden_layers_size * (2 if bidirectional else 1)

        # Instantiate Decoder with Attention
        decoder = self.decoder_class(
            input_size=len(self.output_vocab),
            output_size=len(self.output_vocab),
            layer_type=layer_type,
            emb_dim=emb_size,
            hidden_layers_size=hidden_layers_size,
            num_decod_layers=num_decod_layers,
            dropout_rate=dropout_rate,
            pad_index=self.output_vocab[PAD],
            encoder_hidden_size=encoder_hidden_size,
            bidirectional=bidirectional,
            attention_method=attention_method
        ).to(self.device)

        # Assemble Seq2Seq Model
        seq2seq = self.seq2seq_class(
            encoder=encoder,
            decoder=decoder,
            output_vocab=self.output_vocab
        ).to(self.device)

        optimizer = optim.Adam(seq2seq.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.output_vocab[PAD])

        acc_calculator = AccuracyCalculator(
            eos_token=EOS,
            pad_token=PAD,
            vocab_out=self.output_vocab,
            device=self.device
        )

        train_model = Train_Model(seq2seq, self.train_dataloader, optimizer, criterion, acc_calculator, self.device)
        evaluate_model = Evaluate_Model(seq2seq, self.val_dataloader, criterion, acc_calculator, self.device)

        print(f'The model has {train_model.count_params(seq2seq):,} trainable parameters')

        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Decaying teacher forcing ratio
            decay_rate = 0.05
            current_tfr = teacher_force_ratio * np.exp(-decay_rate * epoch)
            current_tfr = max(0.0, current_tfr)

            print(f'\nEpoch {epoch+1}/{epochs} | Teacher Forcing Ratio: {current_tfr:.4f}\n{"-"*80}')
            train_loss, train_char_acc, train_seq_acc = train_model.train(teacher_force_ratio=current_tfr)
            print(f'Train Loss: {train_loss:.4f} | Char Acc: {train_char_acc:.4f} | Seq Acc: {train_seq_acc:.4f}')

            val_loss, val_char_acc, val_seq_acc = evaluate_model.evaluate(beam_search=val_beam_search, beam_width=beam_width)
            print(f'Val   Loss: {val_loss:.4f} | Char Acc: {val_char_acc:.4f} | Seq Acc: {val_seq_acc:.4f}')

            if wandb_log:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_char_acc': train_char_acc,
                    'train_seq_acc': train_seq_acc,
                    'val_loss': val_loss,
                    'val_char_acc': val_char_acc,
                    'val_seq_acc': val_seq_acc
                })

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(seq2seq.state_dict())
                best_train_loss = train_loss
                best_train_char_acc = train_char_acc
                best_train_seq_acc = train_seq_acc
                best_val_char_acc = val_char_acc
                best_val_seq_acc = val_seq_acc
                epochs_no_improve = 0
                print("Validation improved.")
            else:
                epochs_no_improve += 1
                print(f'No improvement. Patience counter: {epochs_no_improve}/{patience}')
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered.')
                    break

            epoch_mins, epoch_secs = self.epoch_time(start_time, time.time())
            print(f"Epoch Time: {epoch_mins}m {epoch_secs}s")

        # Save the best model
        if best_model_state is not None:
            print("Saving best model...")
            torch.save(best_model_state, save_path)
        else:
            print("Saving last model (no improvement).")
            torch.save(seq2seq.state_dict(), save_path)

        # ========== TEST PHASE ==========
        if testing_phase:
            if not test_path:
                raise ValueError("Test path must be provided when testing_phase=True.")
            
            print(f'\n\n{"+"*28}<Testing Phase Started>{"+"*28}\nPreparing test dataset...\n')
            test_processor = self.sequence_data_preprocessor(test_path, input_vocab=self.input_vocab, output_vocab=self.output_vocab)
            test_input_tensor, test_target_tensor, _, _ = test_processor.prepare_tensors()

            test_data = Datasets(test_input_tensor, test_target_tensor)
            test_dataloader = DataLoader(test_data, batch_size=self.batch_size)

            # Load the best model and evaluate
            best_seq2seq = self.seq2seq_class(
                encoder=encoder,
                decoder=decoder,
                output_vocab=self.output_vocab
            ).to(self.device)
            best_seq2seq.load_state_dict(torch.load(save_path))
            best_seq2seq.eval()

            test_eval_model = Evaluate_Model(
                seq2seq=best_seq2seq,
                dataloader=test_dataloader,
                loss_fn=criterion,
                acc_calculator=acc_calculator,
                device=self.device
            )

            test_loss, test_char_acc, test_seq_acc = test_eval_model.evaluate(
                beam_search=test_beam_search, 
                beam_width=beam_width
            )
            print(f'Test Loss: {test_loss:.4f} | Test Char Acc: {test_char_acc:.4f} | Test Seq Acc: {test_seq_acc:.4f}')

        # Final log return
        log = {
            'train_loss': best_train_loss,
            'train_char_acc': best_train_char_acc,
            'train_seq_acc': best_train_seq_acc,
            'val_loss': best_val_loss,
            'val_char_acc': best_val_char_acc,
            'val_seq_acc': best_val_seq_acc
        }

        if testing_phase:
            log.update({
                'test_loss': test_loss, 'test_char_acc': test_char_acc, 'test_seq_acc': test_seq_acc
            })

        model_params = {
            "input_vocab": self.input_vocab,
            "output_vocab": self.output_vocab,
            "emb_size": emb_size,
            "layer_type": layer_type,
            "hidden_layers_size": hidden_layers_size,
            "num_encod_layers": num_encod_layers,
            "num_decod_layers": num_decod_layers,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "teacher_force_ratio": teacher_force_ratio,
            "bidirectional": bidirectional,
            "beam_width": beam_width,
            "attention_method": attention_method,
            "patience": patience,
            "save_path": save_path
        }

        return seq2seq, train_model, evaluate_model, log, model_params
    
# Example usage
if __name__ == "__main__":

    device = torch.device("cpu")
        
    # Set beam width (1 = greedy decoding, >1 = beam search)

    # Initialize the model with attention
    model = Build_Model(
        sequence_data_preprocessor=SequenceDataPreprocessor,
        encoder_class=Encoder,
        decoder_class=Decoder,
        seq2seq_class=Sequence2Sequence,
        attention_class=Attention,  # <-- Added attention class
        batch_size=32,
        train_path=train_df,
        val_path=dev_df,
        device=device
    )

    # Build and train the model
    seq2seq, train_model, evaluate_model, loss_acc_logs, _ = model.build(
        emb_size= 64,
        layer_type="gru",  # Options: "rnn", "lstm", "gru"
        hidden_layers_size=64,
        num_encod_layers=1,
        num_decod_layers=1,
        dropout_rate=0.3,
        epochs=1,
        learning_rate=0.0001,
        teacher_force_ratio= 1,
        bidirectional= True,
        patience=3,
        val_beam_search= True,  # Uses beam search if beam_width > 1
        beam_width= 1,
        testing_phase=True,          # Set to True if test phase is needed
        test_path=test_df,
        test_beam_search=True,
        wandb_log=False,
        attention_method= "Luong_general"  # Options: "Luong_general", "Bahdanau_concat", "Luong_dot"
    )
