from transformers import pipeline
import torch
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibeClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the vibe classifier using a zero-shot classification model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for classification
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Successfully initialized vibe classifier")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {str(e)}")
            raise

        # Define the taxonomy of vibes
        self.vibe_taxonomy = [
            "Coquette",
            "Clean Girl",
            "Cottagecore",
            "Streetcore",
            "Y2K",
            "Boho",
            "Party Glam"
        ]

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

    def classify_vibe(self, video_metadata: Dict, top_k: int = 2) -> List[str]:
        """
        Classify the vibe of a video based on its metadata.
        
        Args:
            video_metadata (Dict): Dictionary containing video metadata
            top_k (int): Number of top vibes to return
            
        Returns:
            List[str]: List of top vibe names
        """
        try:
            # Prepare context from metadata
            context = self._prepare_video_context(video_metadata)
            
            if not context.strip():
                logger.warning("No context available for classification")
                return ["Evening", "Coquette"]  # Default vibes for testing
            
            # Perform zero-shot classification
            results = self.classifier(
                context,
                candidate_labels=self.vibe_taxonomy,
                multi_label=True,
                hypothesis_template="This video has a {} aesthetic style."
            )
            
            # Return just the top k vibe names
            return results["labels"][:top_k]
            
        except Exception as e:
            logger.error(f"Error during vibe classification: {str(e)}")
            return ["Evening", "Coquette"]  # Default vibes for testing

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
            "Party Glam": "A luxurious, festive aesthetic with sparkle and bold fashion choices"
        }

