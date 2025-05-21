import numpy as np
import pandas as pd
import torch
import random
import time
import math
import csv
from PIL import Image
import io
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import copy
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

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
        self.bidirectional = bidirectional  # Store bidirectional flag
        self.num_encod_layers = num_encod_layers

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
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, layer_type, emb_dim, hidden_layers_size,
                 num_decod_layers, dropout_rate, pad_index, bidirectional=False):
        super().__init__()
        self.layer_type = layer_type.lower()
        self.bidirectional = bidirectional  # Store bidirectional flag
        self.num_layers = num_decod_layers

        self.embed = nn.Embedding(input_size, emb_dim, padding_idx=pad_index)

        rnn_cls = self.layer_mode(self.layer_type)
        self.layer = rnn_cls(
            emb_dim,
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

    def forward(self, inputs: torch.LongTensor, hidden, cell=None):
        inputs = inputs.unsqueeze(1)
        embed = self.embed(inputs)
        
        if self.layer_type == "lstm":
            outputs, (hidden, cell) = self.layer(embed, (hidden, cell))
        else:
            outputs, hidden = self.layer(embed, hidden)
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

        hidden, cell = self.encoder(input_sequence)
        
        # Calculate required dimensions
        encoder_directions = 2 if self.encoder.bidirectional else 1
        decoder_directions = 2 if self.decoder.bidirectional else 1
        encoder_total = self.encoder.num_encod_layers * encoder_directions
        decoder_total = self.decoder.num_layers * decoder_directions

        # Adjust hidden states
        hidden = self.adjust_hidden(hidden, decoder_total)
        cell = self.adjust_hidden(cell, decoder_total) if cell is not None else None

        x = target_sequence[:, 0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output
            x = target_sequence[:, t] if random.random() < teacher_force_ratio else output.argmax(1)
            
        return outputs

    def beam_search_decode(self, input_sequence, sos_token, eos_token, beam_width=3, max_len=30):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            device = input_sequence.device
            hidden, cell = self.encoder(input_sequence)

            # Calculate required dimensions
            encoder_directions = 2 if self.encoder.bidirectional else 1
            decoder_directions = 2 if self.decoder.bidirectional else 1
            encoder_total = self.encoder.num_encod_layers * encoder_directions
            decoder_total = self.decoder.num_layers * decoder_directions

            # Adjust hidden states
            hidden = self.adjust_hidden(hidden, decoder_total)
            cell = self.adjust_hidden(cell, decoder_total) if cell is not None else None

            beams = [([sos_token], 0.0, hidden, cell)]
            completed_sequences = []

            for _ in range(max_len):
                temp_beams = []
                for seq, score, h, c in beams:
                    if seq[-1] == eos_token:
                        completed_sequences.append((seq, score))
                        continue

                    last_token = torch.LongTensor([seq[-1]]).to(device)
                    out, h_new, c_new = self.decoder(last_token, h, c)
                    log_probs = torch.log_softmax(out, dim=1)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        token = top_indices[0][i].item()
                        new_seq = seq + [token]
                        new_score = score + top_log_probs[0][i].item()
                        temp_beams.append((new_seq, new_score, h_new, c_new))

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
        vocab_out:  token->index mapping for your output vocab
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

    def train(self, teacher_force_ratio=0):
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

            # Ignore <sos> token when evaluating
            predicted_trimmed = predicted[:, 1:]
            target_trimmed = target_batch[:, 1:]

            # Flatten the output and target for loss calculation
            output_flatten = output.view(-1, output.shape[-1])
            target_flatten = target_batch.reshape(-1).to(self.device)

            loss = self.loss_fn(output_flatten, target_flatten)
            acc = self.acc_calculator.compute_accuracy(predicted_trimmed, target_trimmed)

            loss.backward()
            # Add gradient clipping here
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

        return epoch_loss / len(self.dataloader), avg_char_acc, avg_seq_acc


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
                    # Beam search decoding
                    batch_size = input_batch.size(0)
                    decoded_batch = []
                    for i in range(batch_size):
                        predicted_ids = self.seq2seq.beam_search_decode(
                            input_batch[i].unsqueeze(0),
                            sos_token=self.seq2seq.output_vocab[SOS],
                            eos_token=self.seq2seq.output_vocab[EOS],
                            beam_width=beam_width
                        )
                        decoded_batch.append(torch.tensor(predicted_ids, device=self.device))
                    
                    # Pad decoded sequences for comparison
                    max_len = max(len(seq) for seq in decoded_batch)
                    predicted_tensor = torch.full((batch_size, max_len), fill_value=self.seq2seq.output_vocab[PAD], device=self.device)
                    for i, seq in enumerate(decoded_batch):
                        predicted_tensor[i, :len(seq)] = seq
                else:
                    output = self.seq2seq(input_batch, target_batch, teacher_force_ratio=0)
                    _, predicted = torch.max(output, dim=2)
                    predicted_tensor = predicted

                # --- Fix for length mismatch between prediction and target ---
                predicted_tensor = predicted_tensor[:, :target_batch.size(1)]
                if predicted_tensor.size(1) < target_batch.size(1):
                    pad_len = target_batch.size(1) - predicted_tensor.size(1)
                    pad = torch.full((predicted_tensor.size(0), pad_len), self.seq2seq.output_vocab[PAD], device=self.device)
                    predicted_tensor = torch.cat([predicted_tensor, pad], dim=1)
                # -------------------------------------------------------------

                # Compute loss
                output = self.seq2seq(input_batch, target_batch, teacher_force_ratio=0)
                output_flat = output.view(-1, output.shape[-1])
                target_flat = target_batch.view(-1)

                loss = self.loss_fn(output_flat, target_flat)

                # Accuracy
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

        return epoch_loss / len(self.dataloader), avg_char_acc, avg_seq_acc


class Build_Model:
    def __init__(self, 
                 sequence_data_preprocessor,
                 encoder, 
                 decoder, 
                 seq2seq, 
                 batch_size,
                 train_path, 
                 val_path,
                 device):
        
        self.sequence_data_preprocessor = sequence_data_preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.seq2seq = seq2seq
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.device = device

        # Process datasets
        train_processor = self.sequence_data_preprocessor(self.train_path)
        self.train_input_tensor, self.train_target_tensor, self.input_vocab, self.output_vocab = train_processor.prepare_tensors()

        # Pass shared vocab to val/test processors
        val_processor = self.sequence_data_preprocessor(self.val_path, input_vocab=self.input_vocab, output_vocab=self.output_vocab)
        self.val_input_tensor, self.val_target_tensor, _, _ = val_processor.prepare_tensors()

        # Create Datasets
        train_data = Datasets(self.train_input_tensor, self.train_target_tensor)
        val_data = Datasets(self.val_input_tensor, self.val_target_tensor)

        # Create DataLoader
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

    def epoch_time(self, start_time, end_time):
        e_time = end_time - start_time
        mins = e_time // 60
        secs = e_time % 60
        return mins, secs

    def build(
        self, emb_size, layer_type, hidden_layers_size, num_encod_layers, num_decod_layers,
        dropout_rate, epochs, learning_rate, teacher_force_ratio=0, bidirectional=False,
        save_path='best_model.pt', patience=3, val_beam_search=False, beam_width=3,
        testing_phase=False, test_path=None, test_beam_search=False
    ):
        # Instantiate encoder, decoder, seq2seq
        encoder = self.encoder(
            input_size=len(self.input_vocab),
            layer_type=layer_type,
            emb_dim=emb_size,
            hidden_layers_size=hidden_layers_size,
            num_encod_layers=num_encod_layers,
            dropout_rate=dropout_rate,
            pad_index=self.input_vocab[PAD],
            bidirectional=bidirectional
        ).to(self.device)

        decoder = self.decoder(
            input_size=len(self.output_vocab),
            output_size=len(self.output_vocab),
            layer_type=layer_type,
            emb_dim=emb_size,
            hidden_layers_size=hidden_layers_size,
            num_decod_layers=num_decod_layers,
            dropout_rate=dropout_rate,
            pad_index=self.output_vocab[PAD],
            bidirectional=bidirectional
        ).to(self.device)

        seq2seq = self.seq2seq(encoder, decoder, self.output_vocab).to(self.device)

        optimizer = optim.Adam(seq2seq.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.output_vocab[PAD]).to(self.device)

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

        best_train_loss, best_train_char_acc, best_train_seq_acc = None, None, None
        best_val_char_acc, best_val_seq_acc = None, None

        for epoch in range(epochs):
            start_time = time.time()

            # === Compute teacher forcing ratio (Exponential Decay) ===
            decay_rate = 0.05  # You can tune this (0.03 - 0.1 are typical)
            current_tfr = teacher_force_ratio * np.exp(-decay_rate * epoch)
            current_tfr = max(0.0, current_tfr)  # Prevent going below 0
            print(f'\nEpoch {epoch+1}/{epochs}{" "*40}Teacher Forcing Ratio: {current_tfr:.4f}\n{"-"*80}')

            train_loss, train_char_acc, train_seq_acc = train_model.train(teacher_force_ratio=current_tfr)

            print(f'Train Loss: {train_loss:.4f} | Train Char Acc: {train_char_acc:.4f} | Train Seq Acc: {train_seq_acc:.4f}')

            val_loss, val_char_acc, val_seq_acc = evaluate_model.evaluate(beam_search=val_beam_search, beam_width=beam_width)
            print(f'Val   Loss: {val_loss:.4f} | Val Char Acc: {val_char_acc:.4f} | Val Seq Acc: {val_seq_acc:.4f}')

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(seq2seq.state_dict())
                best_train_loss = train_loss
                best_train_char_acc = train_char_acc
                best_train_seq_acc = train_seq_acc
                best_val_char_acc = val_char_acc
                best_val_seq_acc = val_seq_acc
                epochs_no_improve = 0
                print(f"\nValidation improved, but waiting to confirm best over next {patience} epochs...")

            else:
                epochs_no_improve += 1
                print(f"\nNo improvement. Patience: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    print(f"\nSaving best model from {epoch+1 - patience} epoch(s) ago with val loss: {best_val_loss:.4f}")
                    torch.save(best_model_state, save_path)
                    print(f"Best model saved to: {save_path}")
                    break

            print(f'\nEpoch Time: {epoch_mins}m {epoch_secs}s')

        if best_model_state is not None:
            print(f'\nTraining ended before confirming best model due to patience.\n\n{"+"*24}<Training Ended after {epochs} Epochs>{"+"*24}')
            torch.save(best_model_state, save_path)
        else:
            print(f'\nNo improvement observed. Saving final model.\n\n{"+"*24}<Training Ended after {epochs} Epochs>{"+"*24}')
            torch.save(seq2seq.state_dict(), save_path)

        # === TESTING PHASE BLOCK ===
        if testing_phase:
            test_path = test_path
            if test_path is None:
                raise ValueError("Test path must be provided for testing_phase=True.")

            print(f'\n\n\n{"+"*28}<Testing Phase Started>{"+"*28}\nPreparing test dataset...')
            test_processor = self.sequence_data_preprocessor(test_path, input_vocab=self.input_vocab, output_vocab=self.output_vocab)
            test_input_tensor, test_target_tensor, _, _ = test_processor.prepare_tensors()
            test_data = Datasets(test_input_tensor, test_target_tensor)
            self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

            print("\nLoading best model for test evaluation...")
            best_seq2seq = self.seq2seq(
                encoder=self.encoder(
                    input_size=len(self.input_vocab),
                    layer_type=layer_type,
                    emb_dim=emb_size,
                    hidden_layers_size=hidden_layers_size,
                    num_encod_layers=num_encod_layers,
                    dropout_rate=dropout_rate,
                    pad_index=self.input_vocab[PAD],
                    bidirectional=bidirectional
                ),
                decoder=self.decoder(
                    input_size=len(self.output_vocab),
                    output_size=len(self.output_vocab),
                    layer_type=layer_type,
                    emb_dim=emb_size,
                    hidden_layers_size=hidden_layers_size,
                    num_decod_layers=num_decod_layers,
                    dropout_rate=dropout_rate,
                    pad_index=self.output_vocab[PAD],
                    bidirectional=bidirectional
                ),
                output_vocab=self.output_vocab
            ).to(self.device)

            best_seq2seq.load_state_dict(torch.load(save_path))
            best_seq2seq.eval()

            test_eval_model = Evaluate_Model(
                seq2seq=best_seq2seq,
                dataloader=self.test_dataloader,
                loss_fn=criterion,
                acc_calculator=acc_calculator,
                device=self.device
            )

            test_loss, test_char_acc, test_seq_acc = test_eval_model.evaluate(beam_search=test_beam_search, beam_width=beam_width)

            print(f'\n Test Loss: {test_loss:.4f} | Test Char Acc: {test_char_acc:.4f} | Test Seq Acc: {test_seq_acc:.4f}\n{"+"*80}')
        else:
            print(f'\n No test evaluation triggered. To evaluate, set `testing_phase=True`.\n{"+"*80}')

        # === RETURN ===
        Loss_accuracy_log = {
            'train_loss': best_train_loss,
            'train_char_acc': best_train_char_acc,
            'train_seq_acc': best_train_seq_acc,
            'val_loss': best_val_loss,
            'val_char_acc': best_val_char_acc,
            'val_seq_acc': best_val_seq_acc
        }

        if testing_phase:
            Loss_accuracy_log.update({
                'test_loss': test_loss,
                'test_char_acc': test_char_acc,
                'test_seq_acc': test_seq_acc
            })

        Log_best_model_params = {
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
            "patience": patience,
            "beam_width": beam_width,
            "save_path": save_path
        }

        return seq2seq, train_model, evaluate_model, Loss_accuracy_log, Log_best_model_params


def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

    # Define file paths
    train_df = '/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_3/predictions_vanilla/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv' 
    dev_df = '/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_3/predictions_vanilla/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv'      
    test_df = '/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_3/predictions_vanilla/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv'    

    # Set beam width (1 = greedy decoding, >1 = beam search)
    beam_width = 3
    beam_search = beam_width > 1

    # Initialize the model
    model = Build_Model(
        sequence_data_preprocessor=SequenceDataPreprocessor,
        encoder=Encoder,
        decoder=Decoder,
        seq2seq=Sequence2Sequence,
        batch_size=32,
        train_path=train_df,
        val_path=dev_df,
        device=device
    )

    # Build and train the model
    seq2seq, train_model, evaluate_model, loss_acc_logs, _ = model.build(
        emb_size=64,
        layer_type="rnn",
        hidden_layers_size=512,
        num_encod_layers=1,
        num_decod_layers=1,
        dropout_rate=0.3,
        epochs=10,
        learning_rate=0.0001,
        teacher_force_ratio=1,
        bidirectional=True,
        patience=3,
        val_beam_search=True,  # Use beam search for validation
        beam_width=3,
        testing_phase=True,    # Set to True for testing
        test_path=test_df,
        test_beam_search=True  # Use beam search for testing
    )

    # Print final results
    print("\nTraining completed!")
    print(f"Best validation loss: {loss_acc_logs['val_loss']:.4f}")
    print(f"Best validation character accuracy: {loss_acc_logs['val_char_acc']:.4%}")
    print(f"Best validation sequence accuracy: {loss_acc_logs['val_seq_acc']:.4%}")
    
    if 'test_loss' in loss_acc_logs:
        print(f"Test loss: {loss_acc_logs['test_loss']:.4f}")
        print(f"Test character accuracy: {loss_acc_logs['test_char_acc']:.4%}")
        print(f"Test sequence accuracy: {loss_acc_logs['test_seq_acc']:.4%}")


if __name__ == "__main__":
    main()
