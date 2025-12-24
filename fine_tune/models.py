import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class KMeansClusteringGPU:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        n_samples = len(X)
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
            self.centroids = X[torch.randint(0, n_samples, (self.n_clusters,), dtype=torch.long)]
        
        for _ in range(self.max_iter):
            distances = torch.cdist(X, self.centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])

            if torch.isnan(new_centroids).any():
                continue

            alpha = self.n_samples_seen_ / (self.n_samples_seen_ + n_samples)
            self.centroids = alpha * self.centroids + (1 - alpha) * new_centroids

            if torch.allclose(self.centroids, new_centroids, atol=1e-6):
                break

        self.n_samples_seen_ += n_samples

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class UNet(nn.Module): 
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

class ECGEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(ECGEmbeddingModel, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEncoding(embedding_dim)
        self.cnn_feature_extractor = UNet(in_channels=1, out_channels=embedding_dim)
    
    def forward(self, tokens, signals):
        tokens = tokens.long()
        token_embedded = self.token_embedding(tokens)
        position_embedded = self.positional_embedding(tokens)
        
        signals = signals.unsqueeze(1)  # [batch_size, seq_len] -> [batch_size, 1, seq_len]
        wave_features = self.cnn_feature_extractor(signals).permute(0, 2, 1)
        
        # 길이 맞춤 (Padding 또는 Slicing)
        seq_len = token_embedded.size(1)
        wave_len = wave_features.size(1)
        
        if wave_len < seq_len:
            wave_features = F.pad(wave_features, (0, 0, 0, seq_len - wave_len))
        elif wave_len > seq_len:
            wave_features = wave_features[:, :seq_len, :]

        combined_embedding = token_embedded + position_embedded + wave_features
        return combined_embedding

class ECGBERTModel(nn.Module):
    def __init__(self, embedding_dim=64, num_layers=12, num_heads=4, dim_feedforward=512, vocab_size=74):
        super(ECGBERTModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        #self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.linear(x)
        #x = self.softmax(x)
        return x
