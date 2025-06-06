from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Optional
import logging
from config import VIBE_CONFIG
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibeClassifier:
    def __init__(self, use_ensemble: bool = VIBE_CONFIG["use_ensemble"]):
        """
        Initialize the vibe classifier using zero-shot classification models.
        
        Args:
            use_ensemble (bool): Whether to use ensemble of models
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Initialize primary classifier (BART)
            self.primary_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize secondary classifier (DistilBERT) if using ensemble
            if use_ensemble:
                self.secondary_classifier = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=0 if self.device == "cuda" else -1
                )
            else:
                self.secondary_classifier = None
            
            logger.info("Successfully initialized vibe classifier(s)")
            
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {str(e)}")
            raise
        
        # Load vibe taxonomy from config
        self.vibe_taxonomy = VIBE_CONFIG["taxonomy"]
        self.confidence_threshold = VIBE_CONFIG["confidence_threshold"]
        self.top_k = VIBE_CONFIG["top_k"]
        self.use_ensemble = use_ensemble
        self.temporal_window = VIBE_CONFIG["temporal_window"]
        
        # Initialize temporal buffer for consistency
        self.temporal_buffer = []

    def _prepare_video_context(self, video_metadata: Dict) -> str:
        """
        Prepare the video context for classification by combining relevant metadata.
        
        Args:
            video_metadata (Dict): Dictionary containing video metadata
            
        Returns:
            str: Combined context string for classification
        """
        context_parts = []
        
        # Add caption if available
        if "caption" in video_metadata:
            context_parts.append(video_metadata["caption"])
            
        # Add description if available
        if "description" in video_metadata:
            context_parts.append(video_metadata["description"])
            
        # Add tags if available
        if "tags" in video_metadata and isinstance(video_metadata["tags"], list):
            context_parts.extend(video_metadata["tags"])
            
        # Combine all parts with spaces
        return " ".join(context_parts)

    def _classify_with_model(self, context: str, classifier: pipeline) -> List[Dict]:
        """
        Classify context using a specific model.
        
        Args:
            context: Text context to classify
            classifier: Huggingface pipeline classifier
            
        Returns:
            List of classification results with scores
        """
        results = classifier(
            context,
            candidate_labels=self.vibe_taxonomy,
            multi_label=True,
            hypothesis_template="This video has a {} aesthetic style."
        )
        
        classifications = []
        for label, score in zip(results["labels"], results["scores"]):
            if score >= self.confidence_threshold:
                classifications.append({
                    "vibe": label,
                    "confidence": float(score)
                })
        
        return classifications

    def _ensemble_classifications(self, primary_results: List[Dict], 
                                secondary_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Combine results from multiple classifiers.
        
        Args:
            primary_results: Results from primary classifier
            secondary_results: Optional results from secondary classifier
            
        Returns:
            Combined and averaged classification results
        """
        if not secondary_results:
            return sorted(primary_results, key=lambda x: x["confidence"], reverse=True)[:self.top_k]
            
        # Combine results
        vibe_scores = {}
        for result in primary_results + secondary_results:
            vibe = result["vibe"]
            confidence = result["confidence"]
            
            if vibe in vibe_scores:
                vibe_scores[vibe].append(confidence)
            else:
                vibe_scores[vibe] = [confidence]
        
        # Average scores
        averaged_results = [
            {"vibe": vibe, "confidence": float(np.mean(scores))}
            for vibe, scores in vibe_scores.items()
        ]
        
        # Sort by confidence and return top-k
        return sorted(averaged_results, key=lambda x: x["confidence"], reverse=True)[:self.top_k]

    def _update_temporal_buffer(self, classifications: List[Dict]):
        """
        Update temporal buffer with new classifications.
        
        Args:
            classifications: New classification results
        """
        self.temporal_buffer.append([c["vibe"] for c in classifications])
        if len(self.temporal_buffer) > self.temporal_window:
            self.temporal_buffer.pop(0)

    def _get_consistent_vibes(self) -> List[str]:
        """
        Get temporally consistent vibes from buffer.
        
        Returns:
            List of most consistent vibe labels
        """
        if not self.temporal_buffer:
            return []
            
        # Flatten all vibes
        all_vibes = [vibe for frame_vibes in self.temporal_buffer for vibe in frame_vibes]
        
        # Count occurrences
        vibe_counts = {}
        for vibe in all_vibes:
            vibe_counts[vibe] = vibe_counts.get(vibe, 0) + 1
        
        # Sort by count and return top-k most consistent vibes
        sorted_vibes = sorted(vibe_counts.items(), key=lambda x: x[1], reverse=True)
        return [vibe for vibe, _ in sorted_vibes[:self.top_k]]

    def classify_vibe(self, video_metadata: Dict) -> List[str]:
        """
        Classify the vibe of a video based on its metadata.
        
        Args:
            video_metadata (Dict): Dictionary containing video metadata
            
        Returns:
            List[str]: List of top vibe names
        """
        try:
            # Prepare context from metadata
            context = self._prepare_video_context(video_metadata)
            
            if not context.strip():
                logger.warning("No context available for classification")
                return []
            
            # Get classifications from primary model
            primary_results = self._classify_with_model(context, self.primary_classifier)
            
            # Get classifications from secondary model if using ensemble
            secondary_results = None
            if self.use_ensemble and self.secondary_classifier:
                secondary_results = self._classify_with_model(context, self.secondary_classifier)
            
            # Combine results
            classifications = self._ensemble_classifications(primary_results, secondary_results)
            
            # Update temporal buffer
            self._update_temporal_buffer(classifications)
            
            # Get temporally consistent vibes
            consistent_vibes = self._get_consistent_vibes()
            
            # If we have consistent vibes, use them; otherwise use current classifications
            if consistent_vibes:
                return consistent_vibes
            else:
                return [c["vibe"] for c in classifications]
            
        except Exception as e:
            logger.error(f"Error during vibe classification: {str(e)}")
            return []

    def get_vibe_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for each vibe in the taxonomy.
        
        Returns:
            Dict[str, str]: Dictionary mapping vibe names to their descriptions
        """
        return {
            "Coquette": "A romantic, feminine aesthetic with soft colors and delicate details",
            "Clean Girl": "A minimalist, fresh aesthetic with neutral tones and natural beauty",
            "Cottagecore": "A rural, nostalgic aesthetic inspired by pastoral life and nature",
            "Streetcore": "An urban, edgy aesthetic with streetwear and city influences",
            "Y2K": "A nostalgic aesthetic from the early 2000s with bold colors and retro elements",
            "Boho": "A free-spirited aesthetic with eclectic patterns and natural elements",
            "Party Glam": "A luxurious, festive aesthetic with sparkle and bold fashion choices",
            "Business Casual": "A polished, professional aesthetic with refined and versatile pieces",
            "Athleisure": "A sporty, comfortable aesthetic blending athletic and leisure wear",
            "Minimalist": "A clean, modern aesthetic focusing on simple lines and neutral colors",
            "Vintage": "A retro-inspired aesthetic drawing from various past decades",
            "Evening": "An elegant, sophisticated aesthetic for formal occasions",
            "Casual Chic": "A relaxed yet stylish aesthetic balancing comfort and fashion",
            "Preppy": "A classic, collegiate-inspired aesthetic with traditional elements",
            "Grunge": "An edgy, rebellious aesthetic with distressed and layered elements"
        }

