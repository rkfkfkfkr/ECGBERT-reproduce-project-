import h5py
import numpy as np
import os
import logging
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import time
import torch.cuda.amp as amp

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Custom Tokenizer 로딩
def get_tokenizer():
    vocab_file_path = "custom_vocab.txt"
    tokenizer = BertTokenizer(vocab_file=vocab_file_path)
    return tokenizer

# ECGDataset 클래스
class ECGDataset(Dataset):
    def __init__(self, hdf5_file, sample_fraction):
        self.hdf5_file = hdf5_file
        self.hdf = h5py.File(self.hdf5_file, 'r')
        self.keys = list(self.hdf.keys())
        self.max_seq_length = self._get_seq_length()

        if sample_fraction < 1.0:
            self.keys = random.sample(self.keys, int(len(self.keys) * sample_fraction))
        logger.info(f"Training with {sample_fraction} - signal_num:{len(self.keys)*12}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        group_key = self.keys[idx]
        group = self.hdf[group_key]

        masked_signal = torch.tensor(group['masked_signal'][:], dtype=torch.float32)
        sentence_token = torch.tensor(group['sentence_token'][:], dtype=torch.long)
        masked_sentence_token = torch.tensor(group['masked_sentence_token'][:], dtype=torch.long)
        masked_sentence_attention_mask = torch.tensor(group['masked_sentence_attention_mask'][:], dtype=torch.long)

        return masked_signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask

    def _get_seq_length(self, sec=10):
        # Sampling frequency for the signals
        return self.hdf[self.keys[0]].attrs['fs']*sec

class UNetCNNEmbedding(nn.Module): 
    def __init__(self, in_channels, embed_dim): 
        super(UNetCNNEmbedding, self).__init__() 
        self.encoder1 = nn.Sequential( 
            nn.Conv1d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(), 
        ) 
        self.encoder2 = nn.Sequential( 
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(embed_dim * 2), 
            nn.ReLU(), 
        ) 
        self.decoder1 = nn.Sequential( 
            nn.ConvTranspose1d(embed_dim * 2, embed_dim, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(), 
        ) 
        self.decoder2 = nn.Sequential( 
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(), 
        ) 
        self.final_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1) 
 
    def forward(self, x): 
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(enc1) 
        dec1 = self.decoder1(enc2) 
        dec2 = self.decoder2(dec1 + enc1) 
 
        x = self.final_conv(dec2) 
 
        return x

class ECGBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers, hidden_dim):
        super(ECGBERT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.cnn_embedding = UNetCNNEmbedding(in_channels=1, embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, masked_sentence_token, sentence_attention_mask, masked_signal):
        batch_size, lead, seq_max_len = masked_sentence_token.size()
        masked_sentence_token_flat = masked_sentence_token.view(batch_size * lead, seq_max_len)
        sentence_attention_mask_flat = sentence_attention_mask.view(batch_size * lead, seq_max_len)
        masked_signal_flat = masked_signal.view(batch_size * lead, 1, seq_max_len)
        
        token_embed = self.token_embedding(masked_sentence_token_flat)
        pos_embed = self.positional_embedding(masked_sentence_token_flat)
        
        masked_signal_embed = self.cnn_embedding(masked_signal_flat).permute(0,2,1)
        
        combined_embed = token_embed + pos_embed + masked_signal_embed
        # [batch_size*lead, seq_max_len, embedd_dim]

        transformer_output = self.transformer(
            combined_embed.permute(1, 0, 2),
            src_key_padding_mask=sentence_attention_mask_flat.bool()
        )

        transformer_output = transformer_output.permute(1, 0, 2)
        logits = self.fc(transformer_output)
        return logits

# Epoch 단위 학습
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    # GradScaler 초기화
    scaler = amp.GradScaler()

    for masked_signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}"):
        masked_signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask = (
            masked_signal.to(device),
            sentence_token.to(device),
            masked_sentence_token.to(device),
            masked_sentence_attention_mask.to(device)
        )

        optimizer.zero_grad()

        # Mixed precision training with autocast
        with amp.autocast():
            logits = model(masked_sentence_token, masked_sentence_attention_mask, masked_signal)
            loss = criterion(logits.view(-1, logits.size(-1)), sentence_token.view(-1))

        # Scaler를 사용한 그라디언트 계산 및 업데이트
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), epoch_time

# Epoch 단위 검증
def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    start_time = time.time()

    with torch.no_grad():
        for masked_signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}/{total_epochs}"):
            masked_signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask = (
                masked_signal.to(device),
                sentence_token.to(device),
                masked_sentence_token.to(device),
                masked_sentence_attention_mask.to(device)
            )

            # Mixed precision inference with autocast
            with amp.autocast():
                logits = model(masked_sentence_token, masked_sentence_attention_mask, masked_signal)
                loss = criterion(logits.view(-1, logits.size(-1)), sentence_token.view(-1))
                total_loss += loss.item()

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), epoch_time

# Custom collate function
def collate_fn(batch, max_seq_length):
    masked_signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask = zip(*batch)

    padded_masked_signals = []
    for signals in masked_signal:
        if signals.size(1) > max_seq_length:
            signals = signals[:, :max_seq_length]  # truncate if longer
        else:
            signals = torch.cat([signals, torch.zeros(signals.size(0), max_seq_length - signals.size(1))], dim=1)  # pad if shorter
        padded_masked_signals.append(signals)

    masked_signal_padded = torch.stack(padded_masked_signals, dim=0)

    sentence_token_padded = torch.stack(sentence_token, dim=0)[:,:,:max_seq_length]
    masked_sentence_token_padded = torch.stack(masked_sentence_token, dim=0)[:,:,:max_seq_length]
    masked_sentence_attention_mask_padded = torch.stack(masked_sentence_attention_mask, dim=0)[:,:,:max_seq_length]

    return masked_signal_padded, sentence_token_padded, masked_sentence_token_padded, masked_sentence_attention_mask_padded


# 모델 학습 메인 함수
def main_train(hdf5_file, model_dir, sample_fraction=1.0, batch_size=8, num_epochs=10, learning_rate=1e-3, device='cuda'):
    tokenizer = get_tokenizer()

    dataset = ECGDataset(hdf5_file, sample_fraction=sample_fraction)
    max_seq_len = dataset._get_seq_length()
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, max_seq_length=max_seq_len))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, max_seq_length=max_seq_len))
    
    vocab_size = len(tokenizer)
    embed_dim = 32
    num_heads = 8
    num_layers = 4
    hidden_dim = 64

    # ECGBERT 모델, 옵티마이저, 스케줄러, 손실 함수 초기화
    model = ECGBERT(vocab_size, embed_dim, max_seq_len, num_heads, num_layers, hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # 2에폭마다 학습률 감소
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 하이퍼파라미터 기반 파일 이름 생성
    file_suffix = f"sf{sample_fraction}_bs{batch_size}_lr{learning_rate}_ep{num_epochs}"
    log_file = os.path.join(model_dir, f"{file_suffix}_training_log.txt")
    model_file = os.path.join(model_dir, f"{file_suffix}_ecgbert_model.pth")
    os.makedirs(model_dir, exist_ok=True)

    # Log train and validation loss
    total_start_time = time.time()

    with open(log_file, "w") as log_f:
        log_f.write(f"Total Epoch: {num_epochs}, data_hdf5: {os.path.basename(hdf5_file)}, sample_fraction: {sample_fraction}, batch_size: {batch_size}, learning_rate: {learning_rate}\n")

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 에폭별 학습 및 검증 수행
            train_loss, train_time = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs)
            val_loss, val_time = validate_epoch(model, val_dataloader, criterion, device, epoch, num_epochs)

            total_time = train_time + val_time
            scheduler.step()  # 스케줄러 업데이트

            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time Taken: {total_time:.2f}s")
            log_f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s\n")

        total_train_time = time.time() - total_start_time
        logger.info(f"Training complete. Total Time Taken: {total_train_time:.2f}s")
        log_f.write(f"Training {num_epochs} epoch, Total Time: {total_train_time:.2f}s\n")

    # 학습된 모델 저장
    torch.save(model.state_dict(), model_file)
    logger.info(f"Model saved at {model_file}.")
    
'''
if __name__ == "__main__":
    dir = "D:/data/ECGBERT/for_git4/preprocessing/"

    sentence_sample_fraction = 0.2
    hdf5_file = os.path.join(dir, f"{sentence_sample_fraction}_sentence_ecg_data.hdf5")

    sample_fraction = 0.001
    model_dir = "D:/data/ECGBERT/for_git4/results/model/"

    # 학습 실행
    main_train(hdf5_file, model_dir, sample_fraction=sample_fraction, batch_size=32, num_epochs=10, learning_rate=1e-4, device='cuda')
'''

def ECGPreTrain(base_dir, output_dir, sentence_sample_fraction, pretrain_sample_fraction=0.001):
    
    hdf5_file = os.path.join(base_dir, f"{sentence_sample_fraction}_sentence_ecg_data.hdf5")
    model_dir = output_dir

    # 학습 실행
    main_train(hdf5_file, model_dir, sample_fraction=pretrain_sample_fraction, batch_size=32, num_epochs=10, learning_rate=1e-4, device='cuda')