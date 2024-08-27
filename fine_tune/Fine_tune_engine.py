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
    
    tokens = [torch.tensor(data[0], device='cuda') for data in batch_data]
    signals = [torch.tensor(data[1], device='cuda') for data in batch_data]
    labels = [torch.tensor(data[2], device='cuda') for data in batch_data]

    tokens = pad_sequences(tokens)
    signals = pad_sequences(signals)
    labels = pad_sequences(labels)
    
    return tokens.float(), signals.float(), labels.float()

class CombinedModel(nn.Module):
    def __init__(self, bert_model, extra_model):
        super(CombinedModel, self).__init__()
        self.bert_model = bert_model
        self.extra_model = extra_model

    def forward(self, x):
        # Forward pass through bert_model
        x = self.bert_model.transformer_encoder(x)
        # Pass the output through extra_model
        x = self.extra_model(x)
        return x

def get_model(save_dir, extra_layers):

    vocab_size = 74  # wave 0~70 + cls + sep + mask , total 74
    embed_size = 256

    bert_model = ECGBERTModel(embed_size).cuda()
    embedding_model = ECGEmbeddingModel(vocab_size, embed_size).cuda()

    extra_model = nn.Sequential()
    for layer in extra_layers:
        extra_model.add_module(layer['name'], layer['module'].cuda())
    
    fine_tune_model = nn.Sequential(
        bert_model,
        extra_model
    )
    
    fine_tune_model_path = os.path.join(save_dir, 'pre_batch_fine_tune_model.pth')
    emb_model_path = os.path.join(save_dir, 'pre_batch_emb_model.pth')

    #fine_tune_model.load_state_dict(torch.load(fine_tune_model_path))
    embedding_model.load_state_dict(torch.load(emb_model_path))
    
    return fine_tune_model, embedding_model


def save_model(fine_tune_model, embedding_model, epoch, batch_num, total_loss, save_dir, all_num_batches):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fine_tune_file_name = f'pre_batch_fine_tune_model.pth'
    emb_file_name = f'pre_batch_emb_model.pth'
    
    if batch_num == all_num_batches-1 :
        fine_tune_file_name = f'fine_tune_model_{epoch+1}_results.pth'
        emb_file_name = f'emb_model_{epoch+1}_results.pth'
    
    fine_tune_model_path = os.path.join(save_dir, fine_tune_file_name)
    emb_model_path = os.path.join(save_dir, emb_file_name)
    torch.save(fine_tune_model.state_dict(), fine_tune_model_path)
    torch.save(embedding_model.state_dict(), emb_model_path)
    
    save_pkl_data(save_dir, f'pre_batch_trotal_loss.pkl', total_loss)

def fine_tune(emb_model, bert_model, experiment, train_data_dir, save_dir):

    extra_model = nn.Sequential()
    for layer in experiment["extra_layers"]:
        extra_model.add_module(layer['name'], layer['module'].cuda())
    fine_tune_model = CombinedModel(bert_model, extra_model)

    fine_tune_model.train()
    emb_model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(bert_model.parameters())) #+ list(extra_model.parameters()), lr=experiment['lr'])

    train_num_batches = len(os.listdir(train_data_dir)) // experiment['batch_size']

    for epoch in range(experiment["epochs"]):
        total_loss = 0
        num_samples = 0
        
        with tqdm(total=train_num_batches, desc=f"Train Epoch {epoch+1}", unit="batch") as pbar:
            for batch_num in range(train_num_batches):

                org_tokens, org_signals, org_labels = get_batch_data(batch_num, experiment["batch_size"], train_data_dir)
                
                if batch_num > 0:
                    _, emb_model = get_model(save_dir, experiment["extra_layers"])
                    fine_tune_file_name = f'pre_batch_fine_tune_model.pth'
                    fine_tune_model_path = os.path.join(save_dir, fine_tune_file_name)
                    fine_tune_model.load_state_dict(torch.load(fine_tune_model_path))
                    emb_model.eval()
                    fine_tune_model.train()

                tokens = org_tokens[:, 1:-1]
                small_batch_seq_len = 10000
                
                for i in range(0, tokens.size(1), small_batch_seq_len):
                    
                    small_tokens = tokens[:, i:i+small_batch_seq_len]
                    small_signals = org_signals[:, i:i+small_batch_seq_len]
                    small_labels = org_labels[:, i:i+small_batch_seq_len]
                    
                    optimizer.zero_grad()
                
                    embeddings = emb_model(small_tokens, small_signals)
                    outputs = fine_tune_model(embeddings).squeeze(-1)
                    outputs = torch.sigmoid(outputs)

                    # Calculate the loss
                    loss = criterion(outputs, small_labels)

                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * small_labels.size(0)
                    num_samples += small_labels.size(0)

                save_model(fine_tune_model, emb_model, epoch, batch_num, total_loss/num_samples, save_dir, train_num_batches)

                pbar.update(1)

                torch.cuda.empty_cache()
        avg_loss = total_loss / num_samples
        logger.info(f'Epoch {epoch+1}/{experiment["epochs"]}, Train Loss: {avg_loss:.4f}')

def evaluate(experiment, val_data_dir, save_dir):
    combined_model, emb_model = get_model(save_dir, experiment["extra_layers"])
    
    combined_model.eval()
    emb_model.eval()
    
    val_num_batches = len(os.listdir(val_data_dir)) // experiment['batch_size']
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(total=val_num_batches, desc="Validation Batches", unit="batch") as pbar:
            for batch_num in range(val_num_batches):
                tokens, signals, labels = get_batch_data(batch_num, experiment["batch_size"], val_data_dir)

                tokens = tokens[:, 1:-1]  # [CLS]와 [SEP] 토큰 제거
                small_batch_seq_len = 10000
                
                batch_preds = []
                batch_labels = []
                
                for i in range(0, tokens.size(1), small_batch_seq_len):
                    small_tokens = tokens[:, i:i+small_batch_seq_len]
                    small_signals = signals[:, i:i+small_batch_seq_len]
                    small_labels = labels[:, i:i+small_batch_seq_len]
                    
                    embeddings = emb_model(small_tokens, small_signals)
                    outputs = combined_model(embeddings).squeeze(-1)
                    preds = torch.sigmoid(outputs)
                    
                    batch_preds.append(preds)
                    batch_labels.append(small_labels)
                
                all_preds.append(torch.cat(batch_preds, dim=1).cpu())
                all_labels.append(torch.cat(batch_labels, dim=1).cpu())
                
                pbar.update(1)

    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_labels = torch.cat(all_labels, dim=0).flatten()
    
    predicted_labels = (all_preds > 0.5).float()
    
    accuracy = (predicted_labels == all_labels).float().mean().item()
    precision = (predicted_labels * all_labels).sum().item() / (predicted_labels.sum().item() + 1e-8)
    recall = (predicted_labels * all_labels).sum().item() / (all_labels.sum().item() + 1e-8)
    specificity = ((1 - predicted_labels) * (1 - all_labels)).sum().item() / ((1 - all_labels).sum().item() + 1e-8)
    
    
    logger.info(f'Validation Accuracy: {accuracy:.4f}')
    logger.info(f'Validation Precision: {precision:.4f}')
    logger.info(f'Validation Recall: {recall:.4f}')
    logger.info(f'Validation Specificity: {specificity:.4f}')
    
    return accuracy, precision, recall, specificity

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def Fine_tune_engine(downstream_tasks, pre_train_model_dir, dir):
    
    for idx, downstream_task in enumerate(downstream_tasks):
        
        save_dir = os.path.join(dir, f'{downstream_task}/results')
        train_data_dir = os.path.join(dir, f'{downstream_task}/ECG_Sentence/train')
        val_data_dir = os.path.join(dir, f'{downstream_task}/ECG_Sentence/val')
    
        vocab_size = 74  # wave 0~70 + cls + sep + mask , total 74s
        embed_size = 256
        
        emb_model = ECGEmbeddingModel(vocab_size, embed_size).cuda()
        bert_model = ECGBERTModel(embed_size).cuda()
        
        state_dict = torch.load(os.path.join(pre_train_model_dir, 'emb_model_1_results.pth'))
        emb_model.load_state_dict(state_dict)
        
        state_dict = torch.load(os.path.join(pre_train_model_dir, 'bert_model_1_results.pth'))
        bert_model.load_state_dict(state_dict)
        
        experiments = [
            {
                "batch_size": 1,
                "lr": 0.001,
                "epochs": 13,
                "class" : 1,
                "extra_layers": [
                    {'name': 'fc1', 'module': nn.Linear(embed_size, vocab_size)},
                    {'name': 'fc2', 'module': nn.Linear(vocab_size, 1)}
                ]
            }
        ]
        
        # label 해결
        
        logger.info(f"Running {downstream_task}")
        fine_tune(emb_model, bert_model, experiments[idx], train_data_dir, save_dir)
            
        # 저장된 모델 로드
        logger.info(f"{downstream_task} Fine Tuning Results")
        accuracy, precision, recall, specificity = evaluate(experiments[idx], val_data_dir, save_dir)