#!/usr/bin/env python3

import sys
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models import Stop
from src.cache import get_or_compute
from src import config


def main():
    stops_file = config.STOP_SETS_DIR / "mumbai_10.json"
    
    with open(stops_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    stops = [Stop(**item) for item in data["stops"]]
    
    print("Computing/loading haversine matrix...")
    haversine_dm = get_or_compute(stops, source='haversine')
    
    google_dm = None
    try:
        # Check if API key is present
        config.get_api_key()
        print("Computing/loading Google Maps matrix...")
        google_dm = get_or_compute(stops, source='google_maps')
    except Exception as e:
        warnings.warn(f"Falling back to haversine only mode. Error: {e}")
        
    print(f"\nMatrix size: {len(stops)}x{len(stops)}")
    print("Haversine DM:")
    print(f"  Min travel time: {haversine_dm.matrix.min():.2f}s")
    print(f"  Max travel time: {haversine_dm.matrix.max():.2f}s")
    print(f"  Mean travel time: {haversine_dm.matrix.mean():.2f}s")
    
    if google_dm is not None:
        print("\nGoogle Maps DM:")
        print(f"  Min travel time: {google_dm.matrix.min():.2f}s")
        print(f"  Max travel time: {google_dm.matrix.max():.2f}s")
        print(f"  Mean travel time: {google_dm.matrix.mean():.2f}s")
        
        ratio_matrix = np.divide(
            google_dm.matrix, 
            haversine_dm.matrix, 
            out=np.zeros_like(google_dm.matrix), 
            where=haversine_dm.matrix != 0
        )
        
        np.fill_diagonal(ratio_matrix, 0)
        
        ratios = ratio_matrix[ratio_matrix != 0]
        if len(ratios) > 0:
            print("\nGoogle/Haversine Ratios:")
            print(f"  Min ratio: {ratios.min():.2f}")
            print(f"  Max ratio: {ratios.max():.2f}")
            print(f"  Mean ratio: {ratios.mean():.2f}")
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        im1 = axes[0].imshow(google_dm.matrix / 60, cmap="YlOrRd")
        axes[0].set_title("Google Maps (minutes)")
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(haversine_dm.matrix / 60, cmap="YlOrRd")
        axes[1].set_title("Haversine (minutes)")
        fig.colorbar(im2, ax=axes[1])
        
        labels = [str(s.id) for s in stops]
        for ax in axes:
            ax.set_xticks(range(len(stops)))
            ax.set_yticks(range(len(stops)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            
        plt.tight_layout()
        save_path = config.PLOTS_DIR / "phase1_heatmaps.png"
        fig.savefig(save_path)
        print(f"\nSaved heatmaps to {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
