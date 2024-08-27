import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from models import ECGEmbeddingModel, ECGBERTModel
from tqdm import tqdm

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def load_batch_data(data_dir, batch_indices):
    batch_data = []
    for idx in batch_indices:
        file_path = os.path.join(data_dir, f'sentence_{idx}.pkl')
        batch_data.append(load_pkl_data(file_path))
    return batch_data

def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    padded_seqs = [torch.cat([seq, torch.zeros(max_len - len(seq), device=seq.device)]) for seq in sequences]
    return torch.stack(padded_seqs)

def get_batch_data(batch_num, batch_size, data_dir):
    batch_indices = range(batch_num * batch_size, (batch_num + 1) * batch_size)
    batch_data = load_batch_data(data_dir, batch_indices)
    
    org_tokens = [torch.tensor(data[0], device='cuda') for data in batch_data]
    masked_tokens = [torch.tensor(data[2], device='cuda') for data in batch_data]
    masked_signals = [torch.tensor(data[3], device='cuda') for data in batch_data]

    org_tokens = pad_sequences(org_tokens)
    masked_tokens = pad_sequences(masked_tokens)
    masked_signals = pad_sequences(masked_signals)
    
    return org_tokens.float(), masked_tokens.float(), masked_signals.float()

def get_model(save_dir, epoch, batch_num):
    
    vocab_size = 74  # wave 0~70 + cls + sep + mask , total 74
    embed_size = 256

    bert_model = ECGBERTModel(embed_size).cuda()
    embedding_model = ECGEmbeddingModel(vocab_size, embed_size).cuda()
    
    bert_model_path = os.path.join(save_dir, f'pre_batch_bert_model.pth')
    emb_model_path = os.path.join(save_dir, f'pre_batch_emb_model.pth')
    bert_model.load_state_dict(torch.load(bert_model_path))
    embedding_model.load_state_dict(torch.load(emb_model_path))
    
    return bert_model, embedding_model

def save_model(bert_model, embedding_model, epoch, batch_num, total_loss, save_dir, all_num_batches):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    bert_file_name = f'pre_batch_bert_model.pth'
    emb_file_name = f'pre_batch_emb_model.pth'
    
    if batch_num == all_num_batches-1 :
        bert_file_name = f'bert_model_{epoch+1}_results.pth'
        emb_file_name = f'emb_model_{epoch+1}_results.pth'
    
    bert_model_path = os.path.join(save_dir, bert_file_name)
    emb_model_path = os.path.join(save_dir, emb_file_name)
    torch.save(bert_model.state_dict(), bert_model_path)
    torch.save(embedding_model.state_dict(), emb_model_path)
    
    save_pkl_data(save_dir, f'pre_batch_trotal_loss.pkl', total_loss)
    

def train_model(save_dir, bert_model, embedding_model, train_data_dir, val_data_dir, train_num_batches, val_num_batches, batch_size, num_epochs, vocab_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(bert_model.parameters(), lr=1e-3)

    embedding_model.train()
    bert_model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        with tqdm(total=train_num_batches, desc=f"Train Epoch {epoch+1}", unit="batch") as pbar:
            for batch_num in range(train_num_batches):

                org_tokens, masked_tokens, masked_signals = get_batch_data(batch_num, batch_size, train_data_dir)
                
                if batch_num > 0:
                    bert_model, embedding_model = get_model(save_dir, epoch, batch_num-1)
                    embedding_model.train()
                    bert_model.train()
                
                optimizer.zero_grad()

                embeddings = embedding_model(masked_tokens, masked_signals)
                outputs = bert_model(embeddings)
                logits = outputs.view(-1, vocab_size)

                loss = criterion(logits, org_tokens.view(-1).long())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                save_model(bert_model, embedding_model, epoch, batch_num, total_loss, save_dir, train_num_batches)

                pbar.update(1)

                torch.cuda.empty_cache()

        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/train_num_batches:.4f}')

        # Validation
        bert_model, embedding_model = get_model(save_dir, epoch, train_num_batches-1)
        embedding_model.eval()
        bert_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(total=val_num_batches, desc="Validation Batches", unit="batch") as pbar:
                for batch_num in range(val_num_batches):
                    org_tokens, masked_tokens, masked_signals = get_batch_data(batch_num, batch_size, val_data_dir)

                    embeddings = embedding_model(masked_tokens, masked_signals)
                    outputs = bert_model(embeddings)
                    logits = outputs.view(-1, vocab_size)

                    val_loss = criterion(logits, org_tokens.view(-1).long())

                    total_val_loss += val_loss.item()
                    pbar.update(1)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss/val_num_batches:.4f}')
        
    return embedding_model, bert_model

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def ECGBERT_Pre_train(dir):
    
    train_data_dir = os.path.join(dir, f'ECG_Sentence/train')
    val_data_dir = os.path.join(dir, f'ECG_Sentence/val')
    save_dir = os.path.join(dir, f'results')
    
    batch_size = 4
    train_num_samples = len(os.listdir(train_data_dir))  # 데이터셋의 총 샘플 수
    val_num_samples = len(os.listdir(val_data_dir)) 
    train_num_batches = train_num_samples // batch_size  # 총 배치 수
    val_num_batches = val_num_samples // batch_size 
    num_epochs = 1
    
    vocab_size = 74  # wave 0~70 + cls + sep + mask , total 74
    embed_size = 256
    
    embedding_model = ECGEmbeddingModel(vocab_size, embed_size).cuda()
    bert_model = ECGBERTModel(embed_size).cuda()

    # 모델 학습
    train_model(save_dir, bert_model, embedding_model, train_data_dir, val_data_dir, train_num_batches, val_num_batches, batch_size, num_epochs, vocab_size)
