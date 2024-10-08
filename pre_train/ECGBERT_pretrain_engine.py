import h5py
import numpy as np
import os
import logging
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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

    def forward(self, masked_sentence_token, sentence_attention_mask, signal):
        batch_size, lead, seq_max_len = masked_sentence_token.size()
        masked_sentence_token_flat = masked_sentence_token.view(batch_size * lead, seq_max_len)
        sentence_attention_mask_flat = sentence_attention_mask.view(batch_size * lead, seq_max_len)
        #logger.info(f"{masked_sentence_token.size()}")
        #logger.info(f"{sentence_attention_mask.size()}")
        #logger.info(f"{signal.size()}")
        signal_flat = signal.view(batch_size * lead, 1, seq_max_len)
        
        token_embed = self.token_embedding(masked_sentence_token_flat)
        pos_embed = self.positional_embedding(masked_sentence_token_flat)
        
        signal_embed = self.cnn_embedding(signal_flat).permute(0,2,1)
        
        combined_embed = token_embed + pos_embed + signal_embed
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

    for signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}"):
        signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask = (
            signal.to(device),
            sentence_token.to(device),
            masked_sentence_token.to(device),
            masked_sentence_attention_mask.to(device)
        )

        optimizer.zero_grad()

        # Mixed precision training with autocast
        with amp.autocast():
            logits = model(masked_sentence_token, masked_sentence_attention_mask, signal)
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
        for signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}/{total_epochs}"):
            signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask = (
                signal.to(device),
                sentence_token.to(device),
                masked_sentence_token.to(device),
                masked_sentence_attention_mask.to(device)
            )

            # Mixed precision inference with autocast
            with amp.autocast():
                logits = model(masked_sentence_token, masked_sentence_attention_mask, signal)
                loss = criterion(logits.view(-1, logits.size(-1)), sentence_token.view(-1))
                total_loss += loss.item()

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), epoch_time

# Custom collate function
def collate_fn(batch, max_seq_length):
    signal, sentence_token, masked_sentence_token, masked_sentence_attention_mask = zip(*batch)

    #logger.info(f"{masked_sentence_token}")# 5, 12, 
    #logger.info(f"{len(masked_sentence_token[0])}")#12
    #logger.info(f"{len(masked_sentence_token[0][0])}")

    padded_lead_signals = []
    for lead_signal in signal:
        if lead_signal.size(1) > max_seq_length:
            lead_signal = lead_signal[:, :max_seq_length]  # truncate if longer
        else:
            lead_signal = torch.cat([lead_signal, torch.zeros(lead_signal.size(0), max_seq_length - lead_signal.size(1))], dim=1)  # pad if shorter
        padded_lead_signals.append(lead_signal)

    signal_padded = torch.stack(padded_lead_signals, dim=0)

    sentence_token_padded = torch.stack(sentence_token, dim=0)[:,:,:max_seq_length]
    masked_sentence_token_padded = torch.stack(masked_sentence_token, dim=0)[:,:,:max_seq_length]
    masked_sentence_attention_mask_padded = torch.stack(masked_sentence_attention_mask, dim=0)[:,:,:max_seq_length]

    return signal_padded, sentence_token_padded, masked_sentence_token_padded, masked_sentence_attention_mask_padded


# 모델 학습 메인 함수
def BERT_pretrain(params, dataset, max_seq_len, device):
    
    num_epochs = int(params['train']['num_epochs'])
    batch_size = int(params['train']['batch_size'])
    learning_rate = float(params['train']['learning_rate'])
    model_dir = params['dataset']['output_dir']
    sample_fraction = float(params['train']['pretrain_sample_fraction'])
    
    tokenizer = get_tokenizer()
    
    if sample_fraction < 1.0:
        sample_size = max(int(sample_fraction*len(dataset)),1)
        dataset, _ = random_split(dataset, [sample_size, len(dataset)-sample_size])
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"{sample_fraction} - train_size: {train_size}, val_size: {val_size}")

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
    #scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # 2에폭마다 학습률 감소
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 하이퍼파라미터 기반 파일 이름 생성
    file_suffix = f"sf{sample_fraction}_bs{batch_size}_lr{learning_rate}_ep{num_epochs}"
    log_file = os.path.join(model_dir, f"{file_suffix}_training_log.txt")
    model_file = os.path.join(model_dir, f"{file_suffix}_ecgbert_model.pth")
    os.makedirs(model_dir, exist_ok=True)

    # Log train and validation loss
    total_start_time = time.time()

    with open(log_file, "w") as log_f:
        log_f.write(f"Total Epoch: {num_epochs}, sample_fraction: {sample_fraction}, batch_size: {batch_size}, learning_rate: {learning_rate}\n")

        for epoch in range(num_epochs):
            
            # 에폭별 학습 및 검증 수행
            train_loss, train_time = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs)
            val_loss, val_time = validate_epoch(model, val_dataloader, criterion, device, epoch, num_epochs)

            total_time = train_time + val_time
            #scheduler.step()  # 스케줄러 업데이트

            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time Taken: {total_time:.2f}s")
            log_f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s\n")

        total_train_time = time.time() - total_start_time
        logger.info(f"Training complete. Total Time Taken: {total_train_time:.2f}s")
        log_f.write(f"Training {num_epochs} epoch, Total Time: {total_train_time:.2f}s\n")

    # 학습된 모델 저장
    torch.save(model.state_dict(), model_file)
    logger.info(f"Model saved at {model_file}.")