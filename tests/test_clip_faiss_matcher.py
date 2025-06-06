import unittest
import numpy as np
import faiss
from models.clip_faiss_matcher import determine_match_type, match_crop_to_catalog
from config import MATCHING_CONFIG

class TestCLIPFAISSMatcher(unittest.TestCase):
    def setUp(self):
        # Create a small test index
        self.dimension = 512  # CLIP embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.product_ids = ["test_product_1", "test_product_2"]
        
        # Add two test vectors
        test_vectors = np.random.rand(2, self.dimension).astype(np.float32)
        self.index.add(test_vectors)
        
    def test_determine_match_type(self):
        # Test exact match
        match_type, conf = determine_match_type(MATCHING_CONFIG["exact_threshold"] + 0.1)
        self.assertEqual(match_type, "exact")
        
        # Test similar match
        match_type, conf = determine_match_type(MATCHING_CONFIG["similar_threshold"] + 0.1)
        self.assertEqual(match_type, "similar")
        
        # Test no match
        match_type, conf = determine_match_type(MATCHING_CONFIG["similar_threshold"] - 0.1)
        self.assertEqual(match_type, "none")
        
    def test_match_crop_to_catalog(self):
        # Create a test embedding
        test_embedding = np.random.rand(self.dimension).astype(np.float32)
        
        # Test matching
        result = match_crop_to_catalog(test_embedding, self.index, self.product_ids)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("matched_product_id", result)
        self.assertIn("match_type", result)
        self.assertIn("confidence", result)
        
        # Check types
        self.assertIsInstance(result["matched_product_id"], str)
        self.assertIsInstance(result["match_type"], str)
        self.assertIsInstance(result["confidence"], float)

if __name__ == '__main__':
    unittest.main() 