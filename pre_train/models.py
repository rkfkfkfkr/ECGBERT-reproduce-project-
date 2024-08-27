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
    def __init__(self, embed_dim, max_len=650002): 
        super().__init__()
        self.P_E = torch.zeros(max_len, embed_dim, device='cuda', requires_grad=False)

        pos = torch.arange(0, max_len, dtype=torch.float, device='cuda').unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device='cuda').float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        self.P_E[:, 0::2] = torch.sin(pos * div_term)
        self.P_E[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len > self.P_E.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.P_E.size(0)}")
        return self.P_E[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.bottleneck = UNetBlock(128, 256)
        self.upconv1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(256, 128)
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool1d(e1, 2))
        b = self.bottleneck(F.max_pool1d(e2, 2))

        d1 = self.upconv1(b)
        d1 = torch.cat([d1, self._crop_tensor(e2, d1)], dim=1)
        d1 = self.dec1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, self._crop_tensor(e1, d2)], dim=1)
        return self.final_conv(self.dec2(d2))

    def _crop_tensor(self, enc_tensor, dec_tensor):
        if enc_tensor.size(2) != dec_tensor.size(2):
            enc_tensor = enc_tensor[:, :, :dec_tensor.size(2)]
        return enc_tensor

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
