import json
import math
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def main():
    print("Loading base model: all-mpnet-base-v2...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled_pairs.json')
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    train_examples = []
    for item in data:
        # Convert label to float for CosineSimilarityLoss (0.0 or 1.0)
        train_examples.append(InputExample(texts=[item['co'], item['po']], label=float(item['label'])))
        
    print(f"Loaded {len(train_examples)} training examples.")
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    epochs = 3
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)

    print(f"Starting fine-tuning for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )
    
    save_path = os.path.join(os.path.dirname(__file__), 'finetuned_mpnet')
    model.save(save_path)
    print(f"\n✅ Fine-tuning complete! Model saved to:\n   {save_path}")
    print("The system will now automatically use this fine-tuned model.")

if __name__ == "__main__":
    main()
