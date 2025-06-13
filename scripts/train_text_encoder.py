# Standard library
import os
import argparse

# Third-party libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from transformers import AutoModel, AutoTokenizer

# Local application (project) imports
from utils.dataframe_utils import process_dataframe, split_data
from utils.pipeline import batch_generator
from utils.text_encoder import TextEncoder

def main(args):

    # Constants
    TEXT_MODEL_NAME = args.text_model_name
    TEXT_MODEL_FROZEN_LAYERS = args.frozen_layers
    EMBED_DIM = args.embed_dim

    VAL_FRAC = args.val_frac
    TRAIN_FRAC = args.train_frac

    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    NUM_EPOCHS = args.num_epochs
    SAVE_PATH = args.save_path
    MODELS_PATH = args.models_path
    CUDA_DEVICE = args.cuda_device

    DEVICE = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('../data/bahar_data_1.csv', low_memory=False)
    process_dataframe(df, inplace=True, verbose=True)

    df_train, df_val, _ = split_data(df, TRAIN_FRAC, VAL_FRAC, verbose=True)

    print("Finished splitting data")

    # === Load Echo Encoder ===
    checkpoint = torch.load(os.path.join(MODELS_PATH, "echo_prime_encoder.pt"), map_location=DEVICE)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, EMBED_DIM)
    echo_encoder.load_state_dict(checkpoint)
    echo_encoder.eval()
    echo_encoder.to(DEVICE)

    # Freeze the Echo Encoder
    for param in echo_encoder.parameters():
        param.requires_grad = False

    text_encoder_full = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    for layer in text_encoder_full.encoder.layer[:TEXT_MODEL_FROZEN_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_encoder = TextEncoder(text_encoder_full, EMBED_DIM).to(DEVICE)

    # === Loss & Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(text_encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # === Training Loop ===
    best_val_loss = float('inf')

    print("Starting training")

    for epoch in range(NUM_EPOCHS):
        text_encoder.train()
        total_train_loss = 0

        for batch in batch_generator(df_train, tokenizer):
            video_tensor, input_ids, attention_mask = batch

            video_tensor = video_tensor.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            with torch.no_grad():
                video_embeds = echo_encoder(video_tensor)
                video_embeds = F.normalize(video_embeds, dim=1)

            text_embeds = text_encoder(input_ids, attention_mask)
            text_embeds = F.normalize(text_embeds, dim=1)

            sim_matrix = torch.matmul(text_embeds, video_embeds.T)
            target = torch.arange(BATCH_SIZE).to(DEVICE)
            loss = criterion(sim_matrix, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / 100

        # === Validation ===
        text_encoder.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in batch_generator(df_val, tokenizer):   
                video_tensor, input_ids, attention_mask = batch
                
                video_tensor = video_tensor.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                video_embeds = echo_encoder(video_tensor)
                video_embeds = F.normalize(video_embeds, dim=1)

                text_embeds = text_encoder(input_ids, attention_mask)
                text_embeds = F.normalize(text_embeds, dim=1)

                sim_matrix = torch.matmul(text_embeds, video_embeds.T)
                target = torch.arange(BATCH_SIZE).to(DEVICE)
                loss = criterion(sim_matrix, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / 20
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(text_encoder.state_dict(), SAVE_PATH)
            print(f"✅ Saved new best model at epoch {epoch + 1} with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    # === CLI Argument Parser ===
    parser = argparse.ArgumentParser(description="Train Text Encoder for Echo-Report Similarity")

    parser.add_argument('--lr', type=float, default=4e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device ID to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_path', type=str, default="../models/best_text_encoder.pth", help='Path to save the best model')
    parser.add_argument('--models_path', type=str, default="../models", help="Path to models")
    parser.add_argument('--train_frac', type=float, default= 39/40, help='Training fraction of the data')
    parser.add_argument('--val_frac', type=float, default=1/400, help='Validation fraction of the data')
    parser.add_argument('--text_model_name', type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", help='Text encoder model name')
    parser.add_argument('--frozen_layers', type=int, default=6, help='Number of frozen layers in the text encoder')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')

    args = parser.parse_args()

    main(args)