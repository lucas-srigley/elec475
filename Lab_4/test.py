import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Import from training script
from train import Config, CLIPModel, COCOCLIPDataset, compute_recall_at_k

# Note: Make sure Config paths match your actual folder structure
# If you get path errors, verify your folder structure matches Config paths

# ==================== Text Query Retrieval ====================
@torch.no_grad()
def retrieve_images_by_text(model, text_query, val_dataset, device, top_k=5):
    """
    Given a text query, retrieve top-k most similar images
    """
    from transformers import CLIPTokenizer, CLIPTextModel
    
    # Load CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained(Config.CLIP_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(Config.CLIP_MODEL).to(device)
    text_encoder.eval()
    
    # Encode query text
    inputs = tokenizer(
        text_query,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    text_outputs = text_encoder(**inputs)
    query_embedding = text_outputs.pooler_output
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Get all image embeddings
    model.eval()
    all_image_embeds = []
    all_image_ids = []
    
    dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    for images, text_embeddings, image_ids in tqdm(dataloader, desc="Computing image embeddings"):
        images = images.to(device)
        image_embeds = model.image_encoder(images)
        all_image_embeds.append(image_embeds.cpu())
        all_image_ids.extend(image_ids.cpu().numpy())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    
    # Compute similarities
    similarities = (query_embedding.cpu() @ all_image_embeds.T).squeeze(0)
    
    # Get top-k
    top_k_values, top_k_indices = similarities.topk(top_k)
    
    # Get corresponding images
    results = []
    for idx, score in zip(top_k_indices, top_k_values):
        image_id = all_image_ids[idx]
        entry = val_dataset.data[image_id]
        results.append({
            'image_id': image_id,
            'filename': entry['filename'],
            'caption': entry['caption'],
            'score': score.item()
        })
    
    return results

# ==================== Zero-shot Classification ====================
@torch.no_grad()
def classify_image(model, image_path, class_labels, device):
    """
    Given an image and a list of class labels, classify the image
    """
    from transformers import CLIPTokenizer, CLIPTextModel
    
    # Load CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained(Config.CLIP_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(Config.CLIP_MODEL).to(device)
    text_encoder.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.CLIP_MEAN, std=Config.CLIP_STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Encode image
    model.eval()
    image_embedding = model.image_encoder(image_tensor)
    
    # Encode class labels
    text_embeddings = []
    for label in class_labels:
        inputs = tokenizer(
            label,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        outputs = text_encoder(**inputs)
        text_embed = outputs.pooler_output
        text_embed = F.normalize(text_embed, p=2, dim=1)
        text_embeddings.append(text_embed)
    
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    # Compute similarities
    similarities = (image_embedding @ text_embeddings.T).squeeze(0)
    probs = F.softmax(similarities * 100, dim=0)  # Temperature scaling
    
    # Get predictions
    results = []
    for label, prob in zip(class_labels, probs):
        results.append({
            'label': label,
            'probability': prob.item()
        })
    
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    return results, image

# ==================== Visualization ====================
def visualize_text_retrieval(query, results, image_dir, save_path=None):
    """
    Visualize top-k retrieved images for a text query
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    fig.suptitle(f'Query: "{query}"', fontsize=16, fontweight='bold')
    
    for idx, (ax, result) in enumerate(zip(axes, results)):
        image_path = Path(image_dir) / result['filename']
        image = Image.open(image_path)
        
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Score: {result['score']:.3f}\n{result['caption'][:50]}...",
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_classification(image, results, save_path=None):
    """
    Visualize image classification results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Show probabilities
    labels = [r['label'] for r in results]
    probs = [r['probability'] for r in results]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    bars = ax2.barh(labels, probs, color=colors)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Classification Results', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ==================== Complete Evaluation ====================
def evaluate_model(checkpoint_path, config):
    """
    Complete evaluation of the trained model
    """
    device = config.DEVICE
    
    print(f"Loading model from {checkpoint_path}...")
    model = CLIPModel(embedding_dim=config.EMBEDDING_DIM).to(device)
    
    if checkpoint_path.endswith('.pt'):
        # Load from checkpoint or best model
        if 'best_model' in checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    model.eval()
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD)
    ])
    
    val_dataset = COCOCLIPDataset(
        config.VAL_IMG_DIR,
        Path(config.CACHE_DIR) / 'val_text_embeddings.pt',
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Compute all embeddings
    print("\nComputing embeddings...")
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for images, text_embeddings, _ in tqdm(val_loader):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            image_embeds = model.image_encoder(images)
            text_embeds = F.normalize(text_embeddings, p=2, dim=1)
            
            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = all_image_embeds @ all_text_embeds.T
    
    # Compute Recall@K metrics
    print("\nComputing Recall@K metrics...")
    metrics = compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10])
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
    print("="*50)
    
    return model, val_dataset, metrics

# ==================== Main Evaluation Script ====================
def main():
    config = Config()
    
    # Path to best model
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    # Evaluate model
    model, val_dataset, metrics = evaluate_model(str(checkpoint_path), config)
    
    # Example 1: Text-to-Image Retrieval
    print("\n" + "="*50)
    print("TEXT-TO-IMAGE RETRIEVAL EXAMPLES")
    print("="*50)
    
    queries = ['sport', 'a cat', 'beach', 'food', 'city']
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retrieve_images_by_text(
            model, query, val_dataset, config.DEVICE, top_k=5
        )
        
        # Print results
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.4f} - {result['caption'][:60]}...")
        
        # Visualize
        save_path = Path(config.CHECKPOINT_DIR) / f'retrieval_{query.replace(" ", "_")}.png'
        visualize_text_retrieval(query, results, config.VAL_IMG_DIR, save_path)
    
    # Example 2: Zero-shot Image Classification
    print("\n" + "="*50)
    print("ZERO-SHOT CLASSIFICATION EXAMPLES")
    print("="*50)
    
    # Get a random image from validation set
    import random
    sample_idx = random.randint(0, len(val_dataset) - 1)
    image_id = val_dataset.image_ids[sample_idx]
    entry = val_dataset.data[image_id]
    image_path = Path(config.VAL_IMG_DIR) / entry['filename']
    
    class_labels = [
        'a photo of a person',
        'a photo of an animal',
        'a photo of a landscape',
        'a photo of food',
        'a photo of a vehicle',
        'a photo of a building'
    ]
    
    print(f"\nClassifying image: {entry['filename']}")
    print(f"True caption: {entry['caption']}")
    
    results, image = classify_image(model, image_path, class_labels, config.DEVICE)
    
    print("\nClassification results:")
    for result in results:
        print(f"  {result['label']}: {result['probability']:.4f}")
    
    # Visualize
    save_path = Path(config.CHECKPOINT_DIR) / 'classification_example.png'
    visualize_classification(image, results, save_path)
    
    print("\n" + "="*50)
    print("Evaluation complete! Visualizations saved to checkpoint directory.")
    print("="*50)

if __name__ == "__main__":
    main()