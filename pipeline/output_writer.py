import json
import os
from typing import List, Dict, Any

def write_output(video_id: str, matches: List[Dict[str, Any]], vibes: List[str]) -> None:
    """Write match results and vibes to a JSON file.
    
    Args:
        video_id: ID of the video being processed
        matches: List of match dictionaries with keys:
                - type: str (e.g. "top", "dress")
                - color: str (e.g. "white", "black")
                - matched_product_id: str
                - match_type: str ("exact" or "partial")
                - confidence: float
        vibes: List of 1-3 vibe labels
    """
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Prepare output data
    output = {
        "video_id": video_id,
        "vibes": vibes,
        "products": matches
    }
    
    # Write to JSON file
    output_path = f"outputs/{video_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Wrote results to {output_path}") 