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

def load_batch_data(data_dir, batch_indices):
    batch_data = []
    for idx in batch_indices:
        file_path = os.path.join(data_dir, f'sentence_{idx}.pkl')
        batch_data.append(load_pkl_data(file_path))
    return batch_data

def get_batch_data(batch_num, batch_size, data_dir):
    batch_indices = range(batch_num * batch_size, (batch_num + 1) * batch_size)
    batch_data = load_batch_data(data_dir, batch_indices)
            
    masked_tokens = [data[0] for data in batch_data]
    original_tokens = [data[1] for data in batch_data]
    original_signals = [data[2] for data in batch_data]

    # 텐서로 변환
    masked_tokens = torch.stack(masked_tokens).cuda()
    original_tokens = torch.stack(original_tokens).cuda()
    original_signals = torch.stack(original_signals).cuda()
    
    return masked_tokens, original_tokens, original_signals

def train_model(bert_model, embedding_model, train_data_dir, val_data_dir, train_num_batches, val_num_batches, batch_size, num_epochs, vocab_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(bert_model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        embedding_model.train()
        bert_model.train()
        total_loss = 0
        
        # 모든 배치를 순회하며 학습
        with tqdm(total=train_num_batches, desc="Train Batches", unit="batch") as pbar:
            for batch_num in range(train_num_batches):
                masked_tokens, original_tokens, original_signals = get_batch_data(batch_num, batch_size, train_data_dir)

                optimizer.zero_grad()

                # 모델의 출력 계산
                embeddings = embedding_model(original_signals, masked_tokens)
                outputs = bert_model(embeddings)
                logits = outputs.logits
                
                # 손실 계산
                masked_positions = (masked_tokens != original_tokens).nonzero(as_tuple=True)
                masked_logits = logits[masked_positions]
                masked_labels = original_tokens[masked_positions]
                
                loss = criterion(masked_logits.view(-1, vocab_size), masked_labels.view(-1))
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            pbar.update(1)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/train_num_batches}')
        
        # Validation
        embedding_model.eval()
        bert_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(total=val_num_batches, desc="Val Batches", unit="batch") as pbar:
                for batch_num in range(val_num_batches):
                    masked_tokens, original_tokens, original_signals = get_batch_data(batch_num, batch_size, val_data_dir)
                    
                    embeddings = embedding_model(original_signals, masked_tokens)
                    outputs = bert_model(embeddings)
                    logits = outputs.logits
                    
                    masked_positions = (masked_tokens != original_tokens).nonzero(as_tuple=True)
                    masked_logits = logits[masked_positions]
                    masked_labels = original_tokens[masked_positions]
                    
                    val_loss = criterion(masked_logits.view(-1, vocab_size), masked_labels.view(-1))
                    total_val_loss += val_loss.item()
                    
                pbar.update(1)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss/val_num_batches}')
        
        return embedding_model, bert_model


def bert_pre_train(train_data_dir, val_data_dir, save_dir):
    
    batch_size = 1024
    train_num_samples = len(os.listdir(train_data_dir))  # 데이터셋의 총 샘플 수
    val_num_samples = len(os.listdir(val_data_dir)) 
    train_num_batches = train_num_samples // batch_size  # 총 배치 수
    val_num_batches = val_num_samples // batch_size 
    num_epochs = 1

    # 모델 생성
    vocab_size = 73
    embed_size = 128
    cnn_channels = 64
    cnn_kernel_size = 3
    unet_channels = 128

    embedding_model = ECGEmbeddingModel(vocab_size, embed_size, cnn_channels, cnn_kernel_size, unet_channels).cuda()
    bert_model = ECGBERTModel(embed_size).cuda()

    # 모델 학습
    embedding_model, bert_model = train_model(bert_model, embedding_model, train_data_dir, val_data_dir, train_num_batches, val_num_batches, batch_size, num_epochs, vocab_size)

    best_model_path = os.path.join(save_dir, 'bert_model.pth')
    torch.save(bert_model.state_dict(), best_model_path)
    print(f'Model saved.')


if __name__ == '__main__':
    train_data_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_Vocab/train/'
    val_data_dir = 'D:/data/ECGBERT/for_git3/preprocessing/ECG_Vocab/val/'
    save_dir = 'D:/data/ECGBERT/for_git3/pre_train/results/'
    
    bert_pre_train(train_data_dir, val_data_dir, save_dir)
