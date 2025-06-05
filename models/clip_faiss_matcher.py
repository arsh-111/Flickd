import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from transformers import CLIPProcessor, CLIPModel
import faiss
import pandas as pd
import os
import pickle

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def save_faiss_index(index, product_ids, index_path, ids_path):
    """Save FAISS index and product IDs to disk.
    
    Args:
        index: FAISS index to save
        product_ids: List of product IDs
        index_path: Path to save FAISS index
        ids_path: Path to save product IDs
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    
    # Save product IDs
    with open(ids_path, 'wb') as f:
        pickle.dump(product_ids, f)

def load_faiss_index(index_path, ids_path):
    """Load FAISS index and product IDs from disk.
    
    Args:
        index_path: Path to FAISS index file
        ids_path: Path to product IDs file
        
    Returns:
        index: Loaded FAISS index
        product_ids: List of product IDs
    """
    index = faiss.read_index(index_path)
    with open(ids_path, 'rb') as f:
        product_ids = pickle.load(f)
    return index, product_ids

def get_clip_embedding(image_path):
    """Get CLIP embedding for an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        embedding: 512-dimensional numpy embedding
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get CLIP embedding
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().astype(np.float32)[0]

def load_image(image_url):
    """Load and verify an image from a URL.
    
    Args:
        image_url: URL of the image to load
        
    Returns:
        PIL.Image or None if loading fails
    """
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        image = Image.open(BytesIO(response.content))
        image.verify()  # Verify it's a valid image
        return Image.open(BytesIO(response.content)).convert("RGB")  # Reopen after verify
    except requests.exceptions.RequestException as e:
        print(f"Network error loading {image_url}: {e}")
        return None
    except (IOError, OSError) as e:
        print(f"Image format error for {image_url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {image_url}: {e}")
        return None

def build_faiss_index_from_catalog(images_csv):
    """
    Loads product images from image_url in images.csv, embeds each image, builds and returns a FAISS index and product_ids list.
    If cached index exists, loads it instead of rebuilding.
    
    Args:
        images_csv: Path to images.csv file with columns 'id' and 'image_url'
    Returns:
        index: FAISS index of image embeddings
        product_ids: List of product IDs in same order as index
    """
    # Define cache paths
    index_path = 'models/faiss_index.index'
    ids_path = 'models/product_ids.pkl'
    
    # Check if cached index exists
    if os.path.exists(index_path) and os.path.exists(ids_path):
        print("Loading cached FAISS index and product IDs...")
        return load_faiss_index(index_path, ids_path)
    
    print(f"\nBuilding new index from catalog: {images_csv}...")
    df = pd.read_csv(images_csv)
    total_products = len(df)
    print(f"Found {total_products} products in catalog")
    
    product_ids = []
    embeddings = []
    failed_images = 0
    failed_urls = []
    
    for idx, row in df.iterrows():
        url = row['image_url']
        product_id = row['id']
        
        # Load and verify image
        image = load_image(url)
        if image is None:
            failed_images += 1
            failed_urls.append(url)
            continue
            
        try:
            # Get CLIP embedding
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy().astype(np.float32)[0])
            product_ids.append(product_id)
            
            # Print progress every 10 products
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_products} products...")
                
        except Exception as e:
            print(f"Failed to process product {product_id} from {url}: {e}")
            failed_images += 1
            failed_urls.append(url)
            continue
    
    if not embeddings:
        raise ValueError("No valid embeddings were created from the catalog.")
    
    # Print summary
    print("\nCatalog Processing Summary:")
    print(f"Total products: {total_products}")
    print(f"Successfully processed: {len(embeddings)}")
    print(f"Failed to process: {failed_images}")
    if failed_urls:
        print("\nFailed URLs:")
        for url in failed_urls[:5]:  # Show first 5 failed URLs
            print(f"- {url}")
        if len(failed_urls) > 5:
            print(f"... and {len(failed_urls) - 5} more")
    
    # Create FAISS index using L2 distance
    print("\nBuilding FAISS index...")
    embeddings = np.stack(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
    index.add(embeddings)
    print("FAISS index built successfully")
    
    # Save the index and product IDs
    print("Saving index and product IDs to disk...")
    save_faiss_index(index, product_ids, index_path, ids_path)
    print("Cache saved successfully")
    
    return index, product_ids

def match_crop_to_catalog(crop_embedding, faiss_index, product_ids):
    """Match a crop embedding to the catalog using FAISS.
    
    Args:
        crop_embedding: CLIP embedding of the crop
        faiss_index: FAISS index of catalog embeddings
        product_ids: List of product IDs in same order as index
        
    Returns:
        matched_product_id: ID of best matching product
        match_type: Type of match (exact/partial)
        confidence: Confidence score of match
    """
    # Search for nearest neighbor
    D, I = faiss_index.search(crop_embedding.reshape(1, -1), 1)
    
    # Get match details
    matched_product_id = product_ids[I[0][0]]
    confidence = float(D[0][0])  # Cosine similarity score
    
    # Determine match type based on confidence
    match_type = "exact" if confidence > 0.8 else "partial"
    
    return matched_product_id, match_type, confidence 