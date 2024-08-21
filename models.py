import torch
import torch.nn as nn
import torch.nn.functional as F

class KMeansClusteringGPU:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.n_samples_seen_ = 0  # 추가된 속성

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        #n_samples, n_features = X.shape
        n_samples = len(X)
        
        # 여기서 int 타입의 텐서가 되도록 변환합니다.
        self.centroids = X[torch.randint(0, n_samples, (self.n_clusters,), dtype=torch.long)]

        for _ in range(self.max_iter):
            distances = torch.cdist(X, self.centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])

            if torch.allclose(self.centroids, new_centroids, atol=1e-6):
                break

            self.centroids = new_centroids
            
    def partial_fit(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        n_samples, n_features = X.shape
        
        if self.centroids is None:
            # 초기화
            self.centroids = X[torch.randint(0, n_samples, (self.n_clusters,), dtype=torch.long)]
        
        for _ in range(self.max_iter):
            distances = torch.cdist(X, self.centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])

            # NaN 체크 및 처리
            if torch.isnan(new_centroids).any():
                #print("NaN detected in new_centroids, skipping update.")
                continue

            # 이전 centroid와 새로운 centroid의 차이를 기반으로 업데이트
            if self.n_samples_seen_ > 0:
                alpha = self.n_samples_seen_ / (self.n_samples_seen_ + n_samples)
                self.centroids = alpha * self.centroids + (1 - alpha) * new_centroids
            else:
                self.centroids = new_centroids

            if torch.allclose(self.centroids, new_centroids, atol=1e-6):
                break

        self.n_samples_seen_ += n_samples  # 샘플 수 업데이트

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

class CNNEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNNEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class ECGEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, cnn_channels, cnn_kernel_size, unet_channels):
        super(ECGEmbeddingModel, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size)
        self.cnn_embedding = CNNEmbedding(1, cnn_channels, cnn_kernel_size)
        self.unet_down1 = UNetBlock(cnn_channels, unet_channels)
        self.unet_down2 = UNetBlock(unet_channels, unet_channels)
        self.unet_up1 = nn.ConvTranspose1d(unet_channels, cnn_channels, kernel_size=2, stride=2)
        self.unet_up2 = nn.ConvTranspose1d(cnn_channels, 1, kernel_size=2, stride=2)

    def forward(self, x, token_ids):
        # CNN Feature Extraction
        x = self.cnn_embedding(x)
        x = self.unet_down1(x)
        x = self.unet_down2(x)
        x = self.unet_up1(x)
        cnn_features = self.unet_up2(x)

        # Token and Position Embedding
        token_embed = self.token_embedding(token_ids)
        position_embed = self.position_embedding(token_embed)

        # Combine embeddings
        combined_embedding = cnn_features + position_embed
        return combined_embedding

class ECGBERTModel(nn.Module):
    def __init__(self, embedding_dim, num_layers=12, num_heads=8, dim_feedforward=512, vocab_size = 73):
        super(ECGBERTModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

'''
# Example usage:
vocab_size = 73
embed_size = 128
cnn_channels = 64
cnn_kernel_size = 3
unet_channels = 128

model = ECGEmbeddingModel(vocab_size, embed_size, cnn_channels, cnn_kernel_size, unet_channels)
ecg_signal = torch.randn(1, 1, 500)  # Example ECG signal
token_ids = torch.randint(0, vocab_size, (1, 500))  # Example token IDs

output = model(ecg_signal, token_ids)
print(output.shape)

# ecg_signal 은 preprocessed_signal, 
# token_ids는 sentence
'''