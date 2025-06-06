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
import logging
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional
from config import MATCHING_CONFIG

logger = logging.getLogger(__name__)

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

def extract_dominant_colors(image: Image.Image, n_colors: int = MATCHING_CONFIG["num_colors"]) -> List[str]:
    """Extract dominant colors from an image using k-means clustering.
    
    Args:
        image: PIL Image
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of color names
    """
    # Convert image to RGB array
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_.astype(int)
    
    # Convert RGB values to color names
    color_names = []
    for rgb in colors:
        # Simple color naming based on RGB values
        r, g, b = rgb
        if max(r, g, b) < 64:
            color_names.append("black")
        elif min(r, g, b) > 192:
            color_names.append("white")
        elif r > max(g, b) + 64:
            color_names.append("red")
        elif g > max(r, b) + 64:
            color_names.append("green")
        elif b > max(r, g) + 64:
            color_names.append("blue")
        elif abs(r - g) < 32 and abs(g - b) < 32:
            color_names.append("gray")
        else:
            color_names.append("mixed")
    
    return color_names

def get_clip_embedding(image_path: str) -> np.ndarray:
    """Get CLIP embedding for an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        embedding: 512-dimensional numpy embedding
    """
    try:
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Validate image dimensions
        if image.size[0] < 10 or image.size[1] < 10:
            raise ValueError(f"Image too small: {image.size}")
            
        # Log image info
        logger.debug(f"Processing image {image_path}: size={image.size}, mode={image.mode}")
        
        # Get CLIP embedding
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        # Validate embedding
        embedding_np = embedding.cpu().numpy().astype(np.float32)[0]
        if not np.isfinite(embedding_np).all():
            raise ValueError(f"Invalid embedding values detected for {image_path}")
            
        logger.debug(f"Generated embedding shape={embedding_np.shape}, norm={np.linalg.norm(embedding_np):.3f}")
        return embedding_np
        
    except Exception as e:
        logger.error(f"Error getting CLIP embedding for {image_path}: {str(e)}")
        raise

def determine_match_type(similarity: float) -> Tuple[str, float]:
    """Determine match type and confidence based on similarity score.
    
    Args:
        similarity: Cosine similarity score
        
    Returns:
        tuple: (match_type, confidence)
    """
    if similarity >= MATCHING_CONFIG["exact_threshold"]:
        return "exact", similarity
    elif similarity >= MATCHING_CONFIG["similar_threshold"]:
        return "similar", similarity
    else:
        return "none", similarity

def match_crop_to_catalog(crop_embedding: np.ndarray, faiss_index: faiss.Index, 
                         product_ids: List[str], product_metadata: Dict[str, Dict],
                         image_path: Optional[str] = None, k: int = 5) -> Dict:
    """Match a crop embedding to the catalog using FAISS.
    
    Args:
        crop_embedding: CLIP embedding of the crop
        faiss_index: FAISS index of catalog embeddings
        product_ids: List of product IDs in same order as index
        product_metadata: Dictionary mapping product IDs to their metadata
        image_path: Optional path to crop image for color extraction
        k: Number of top matches to return (default: 5)
        
    Returns:
        dict: Match results including:
            - matched_product_id: ID of best matching product
            - match_type: Type of match (exact/similar/none)
            - confidence: Confidence score of match
            - colors: List of dominant colors (if image_path provided)
            - top_matches: List of top k matches with scores
            - product_type: Type of the matched product
            - product_color: Color of the matched product
            - product_style: Style of the matched product
    """
    try:
        # Validate inputs
        if crop_embedding.shape != (512,):
            raise ValueError(f"Invalid embedding shape: {crop_embedding.shape}, expected (512,)")
        if not np.isfinite(crop_embedding).all():
            raise ValueError("Invalid embedding values detected")
            
        # Normalize embedding if not already normalized
        norm = np.linalg.norm(crop_embedding)
        if abs(norm - 1.0) > 1e-6:
            logger.warning(f"Input embedding not normalized (norm={norm:.3f}). Normalizing...")
            crop_embedding = crop_embedding / norm
            
        # Search for k nearest neighbors
        D, I = faiss_index.search(crop_embedding.reshape(1, -1), k)
        similarities = 1 - D[0]  # Convert L2 distances to similarities
        
        # Log top matches
        logger.info(f"\nTop {k} matches for {image_path if image_path else 'crop'}:")
        top_matches = []
        for idx, (sim, prod_idx) in enumerate(zip(similarities, I[0])):
            match_type, _ = determine_match_type(sim)
            prod_id = str(product_ids[prod_idx])
            metadata = product_metadata.get(prod_id, {})
            logger.info(f"{idx+1}. Product {prod_id}: {match_type} match (similarity: {sim:.3f})")
            logger.info(f"   Type: {metadata.get('type', 'unknown')}, Color: {metadata.get('color', 'unknown')}")
            top_matches.append({
                "product_id": prod_id,
                "similarity": float(sim),
                "match_type": match_type,
                "type": metadata.get('type', 'unknown'),
                "color": metadata.get('color', 'unknown'),
                "style": metadata.get('style', 'unknown')
            })
            
        # Get best match details
        similarity = similarities[0]
        match_type, confidence = determine_match_type(similarity)
        
        # Skip if confidence is too low
        if confidence < MATCHING_CONFIG["min_confidence"]:
            logger.warning(f"Low confidence match ({confidence:.2f}) - skipping")
            return None
            
        # Get metadata for best match
        best_match_id = str(product_ids[I[0][0]])
        best_match_metadata = product_metadata.get(best_match_id, {})
            
        result = {
            "matched_product_id": best_match_id,
            "match_type": str(match_type),
            "confidence": float(confidence),
            "top_matches": top_matches,
            "product_type": best_match_metadata.get('type', 'unknown'),
            "product_color": best_match_metadata.get('color', 'unknown'),
            "product_style": best_match_metadata.get('style', 'unknown')
        }
        
        # Extract colors if image path provided
        if image_path:
            image = Image.open(image_path).convert("RGB")
            colors = extract_dominant_colors(image)
            logger.info(f"Extracted colors: {colors}")
            result["detected_colors"] = colors
        
        return result
        
    except Exception as e:
        logger.error(f"Error matching crop to catalog: {str(e)}")
        raise

def build_faiss_index_from_catalog(images_csv: str, catalog_xlsx: str = "data/catalog.xlsx") -> Tuple[faiss.Index, List[str], Dict[str, Dict]]:
    """Build FAISS index from product catalog.
    
    Args:
        images_csv: Path to images.csv file with columns 'id' and 'image_url'
        catalog_xlsx: Path to catalog.xlsx with product metadata
        
    Returns:
        tuple: (FAISS index, list of product IDs, dict of product metadata)
    """
    # Define cache paths
    index_path = 'models/faiss_index.index'
    ids_path = 'models/product_ids.pkl'
    metadata_path = 'models/product_metadata.pkl'
    
    # Check if cached data exists
    if os.path.exists(index_path) and os.path.exists(ids_path) and os.path.exists(metadata_path):
        logger.info("Loading cached FAISS index, product IDs, and metadata...")
        index = faiss.read_index(index_path)
        with open(ids_path, 'rb') as f:
            product_ids = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            product_metadata = pickle.load(f)
        return index, product_ids, product_metadata
    
    logger.info(f"Building new index from catalog: {images_csv}...")
    
    # Load product metadata
    catalog_df = pd.read_excel(catalog_xlsx)
    product_metadata = {}
    
    for _, row in catalog_df.iterrows():
        # Extract product type and collections
        product_type = row.get('product_type', 'unknown')
        collections = str(row.get('product_collections', '')).split(',')
        collections = [c.strip() for c in collections]
        
        # Extract style from collections (e.g., "Casual Essentials", "Party Wear", etc.)
        style = 'unknown'
        for collection in collections:
            if any(keyword in collection.lower() for keyword in ['casual', 'party', 'formal', 'ethnic', 'western']):
                style = collection
                break
        
        # Extract product tags
        tags = str(row.get('product_tags', '')).split(',')
        tags = [tag.strip() for tag in tags]
        
        # Determine color from tags or title
        color = 'unknown'
        title = str(row.get('title', '')).lower()
        common_colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'brown', 'grey', 'gray', 'navy']
        
        # Try to find color in tags first
        for tag in tags:
            if any(color_name in tag.lower() for color_name in common_colors):
                color = tag.strip()
                break
                
        # If no color found in tags, try title
        if color == 'unknown':
            for color_name in common_colors:
                if color_name in title:
                    color = color_name
                    break
        
        product_metadata[str(row['id'])] = {
            'type': product_type,
            'color': color,
            'style': style,
            'title': row.get('title', ''),
            'description': row.get('description', ''),
            'tags': tags,
            'collections': collections
        }
    
    # Load and process images
    df = pd.read_csv(images_csv)
    total_products = len(df)
    logger.info(f"Found {total_products} products in catalog")
    
    product_ids = []
    embeddings = []
    failed_images = 0
    failed_urls = []
    
    for idx, row in df.iterrows():
        url = row['image_url']
        product_id = str(row['id'])
        
        try:
            # Skip if we don't have metadata for this product
            if product_id not in product_metadata:
                logger.warning(f"No metadata found for product {product_id}, skipping...")
                continue
                
            # Load and verify image
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Get CLIP embedding
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            embeddings.append(embedding.cpu().numpy().astype(np.float32)[0])
            product_ids.append(product_id)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{total_products} products...")
                
        except Exception as e:
            logger.error(f"Failed to process product {product_id} from {url}: {str(e)}")
            failed_images += 1
            failed_urls.append(url)
            continue
    
    if not embeddings:
        raise ValueError("No valid embeddings were created from the catalog.")
    
    # Print summary
    logger.info("\nCatalog Processing Summary:")
    logger.info(f"Total products: {total_products}")
    logger.info(f"Successfully processed: {len(embeddings)}")
    logger.info(f"Failed to process: {failed_images}")
    
    # Create FAISS index
    logger.info("\nBuilding FAISS index...")
    embeddings = np.stack(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save index, IDs and metadata
    logger.info("Saving index, product IDs, and metadata to disk...")
    faiss.write_index(index, index_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(product_ids, f)
    with open(metadata_path, 'wb') as f:
        pickle.dump(product_metadata, f)
    
    return index, product_ids, product_metadata 