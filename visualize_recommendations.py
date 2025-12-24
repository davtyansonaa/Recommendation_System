"""
Visualization Script for Enhanced Fashion Recommendation System
================================================================

This script loads the pre-trained index and generates visual recommendations
WITHOUT retraining the model.


Usage:
    python visualize_recommendations.py

Requirements:
    - Pre-trained index file in ./indexes/enhanced_train_index.pkl
    - matplotlib installed: pip install matplotlib
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from typing import List, Dict
import random


# ==================== Configuration ====================
class Config:
    INDEX_FILE = "./indexes/enhanced_train_index.pkl"
    OUTPUT_DIR = "./recommendations_output"
    DEFAULT_K = 10
    RANDOM_SEED = 42


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
np.random.seed(Config.RANDOM_SEED)
random.seed(Config.RANDOM_SEED)


# ==================== Load Pre-trained Index ====================
def load_pretrained_system():
    """Load the pre-trained recommendation system"""
    
    if not os.path.exists(Config.INDEX_FILE):
        print(f"❌ ERROR: Index file not found: {Config.INDEX_FILE}")
        print("Please run the main training script first to create the index.")
        return None
    
    print(f"[Loading] Reading index from {Config.INDEX_FILE}...")
    
    with open(Config.INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded successfully!")
    print(f"   - Products: {len(data['product_ids'])}")
    print(f"   - Feature dimension: {data['features'].shape[1]}")
    if data.get('reduced_features') is not None:
        print(f"   - Reduced dimension: {data['reduced_features'].shape[1]}")
    
    return data


# ==================== Get Recommendations ====================
def get_recommendations_from_index(data, query_idx: int, k: int = 10) -> List[Dict]:
    """Get recommendations using pre-computed similarity matrix"""
    
    if data.get('similarity_matrix') is None:
        print("[Computing] Similarity matrix not found, computing now...")
        from sklearn.metrics.pairwise import cosine_similarity
        features = data.get('reduced_features', data['features'])
        data['similarity_matrix'] = cosine_similarity(features)
    
    similarities = data['similarity_matrix'][query_idx].copy()
    similarities[query_idx] = -1  # Exclude self
    
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    recommendations = []
    for i, idx in enumerate(top_k_indices):
        rec = {
            'rank': i + 1,
            'image_id': data['product_ids'][idx],
            'similarity': float(similarities[idx]),
            'image_path': data['product_paths'][idx],
            'labels': data['label_info'].get(data['product_ids'][idx], [])
        }
        recommendations.append(rec)
    
    return recommendations


# ==================== Visualization Functions ====================
def visualize_recommendations(data, query_idx, k=10, save_path=None):
    """Create a visual grid of query + recommendations"""
    
    recommendations = get_recommendations_from_index(data, query_idx, k)
    query_id = data['product_ids'][query_idx]
    query_path = data['product_paths'][query_idx]
    query_labels = data['label_info'].get(query_id, [])
    
    # Create figure
    n_cols = min(6, k + 1)
    n_rows = (k + 1 + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3.5))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot query image
    ax = fig.add_subplot(gs[0, 0])
    try:
        query_img = Image.open(query_path).convert('RGB')
        ax.imshow(query_img)
        ax.set_title(f'🔍 QUERY\nID: {query_id}\n',
                    fontweight='bold', fontsize=10, color='red', pad=10)
        ax.axis('off')
        # Red border
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(4)
            spine.set_visible(True)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading\n{query_id}', 
               ha='center', va='center', fontsize=8)
        ax.axis('off')
    
    # Plot recommendations
    for i, rec in enumerate(recommendations):
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        try:
            img = Image.open(rec['image_path']).convert('RGB')
            ax.imshow(img)
            
            # Show label overlap
            overlap = len(set(query_labels) & set(rec['labels']))
            overlap_pct = (overlap / len(query_labels) * 100) if query_labels else 0
            
            title = f"#{rec['rank']}\n"
            title += f"Sim: {rec['similarity']:.3f}\n"
            title += f"Match: {overlap}/{len(query_labels)} ({overlap_pct:.0f}%)"
            
            ax.set_title(title, fontsize=9, pad=8)
            ax.axis('off')
            
            # Green border for recommendations
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
                spine.set_visible(True)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{rec["image_id"]}', 
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
    
    plt.suptitle(f'Fashion Recommendations (K={k})', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_DIR, f'recommendations_{query_id}.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    return save_path


def visualize_multiple_queries(data, n_queries=5, k=10):
    """Generate recommendations for multiple random queries"""
    
    print(f"\n[Visualizing] Generating {n_queries} recommendation sets...")
    
    n_products = len(data['product_ids'])
    random_indices = random.sample(range(n_products), n_queries)
    
    saved_files = []
    for i, query_idx in enumerate(random_indices, 1):
        print(f"\n[{i}/{n_queries}] Query index: {query_idx}")
        query_id = data['product_ids'][query_idx]
        print(f"           Query ID: {query_id}")
        
        save_path = visualize_recommendations(data, query_idx, k=k)
        saved_files.append(save_path)
    
    return saved_files


def visualize_by_image_id(data, image_id: str, k=10):
    """Visualize recommendations for a specific image ID"""
    
    if image_id not in data['product_ids']:
        print(f"❌ ERROR: Image ID '{image_id}' not found in index")
        print(f"Available IDs: {data['product_ids'][:10]}...")
        return None
    
    query_idx = data['product_ids'].index(image_id)
    return visualize_recommendations(data, query_idx, k=k)


def create_comparison_grid(data, query_indices: List[int], k=5):
    """Create a comparison showing multiple queries side-by-side"""
    
    n_queries = len(query_indices)
    fig = plt.figure(figsize=(18, n_queries * 3))
    gs = GridSpec(n_queries, k + 1, figure=fig, hspace=0.3, wspace=0.2)
    
    for row, query_idx in enumerate(query_indices):
        recommendations = get_recommendations_from_index(data, query_idx, k)
        query_id = data['product_ids'][query_idx]
        query_path = data['product_paths'][query_idx]
        
        # Query image
        ax = fig.add_subplot(gs[row, 0])
        try:
            img = Image.open(query_path).convert('RGB')
            ax.imshow(img)
            ax.set_title(f'Query\n{query_id}', fontweight='bold', fontsize=9)
            ax.axis('off')
        except:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
        
        # Recommendations
        for col, rec in enumerate(recommendations, 1):
            ax = fig.add_subplot(gs[row, col])
            try:
                img = Image.open(rec['image_path']).convert('RGB')
                ax.imshow(img)
                ax.set_title(f"{rec['similarity']:.3f}", fontsize=8)
                ax.axis('off')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
    
    plt.suptitle('Multiple Query Comparison', fontsize=14, fontweight='bold')
    save_path = os.path.join(Config.OUTPUT_DIR, 'comparison_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison: {save_path}")
    plt.close()
    
    return save_path


def print_text_recommendations(data, query_idx, k=10):
    """Print recommendations in text format"""
    
    recommendations = get_recommendations_from_index(data, query_idx, k)
    query_id = data['product_ids'][query_idx]
    query_labels = data['label_info'].get(query_id, [])
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n🔍 Query Product:")
    print(f"   ID: {query_id}")
    print(f"   Path: {data['product_paths'][query_idx]}")
    print(f"   Labels: {query_labels}")
    
    print(f"\n📋 Top {k} Recommendations:")
    print("-" * 80)
    
    for rec in recommendations:
        overlap = len(set(query_labels) & set(rec['labels']))
        label_str = str(rec['labels'][:5])
        if len(rec['labels']) > 5:
            label_str = label_str[:-1] + ', ...]'
        
        print(f"  {rec['rank']:2d}. ID: {rec['image_id']:15s} | "
              f"Similarity: {rec['similarity']:.4f} | "
              f"Label overlap: {overlap:2d}/{len(query_labels):2d} | "
              f"Labels: {label_str}")
    
    print("=" * 80)


# ==================== Interactive Mode ====================
def interactive_mode(data):
    """Interactive command-line interface"""
    
    print("\n" + "=" * 80)
    print("INTERACTIVE RECOMMENDATION VIEWER")
    print("=" * 80)
    print("\nCommands:")
    print("  'random' or 'r'          - Show random recommendation")
    print("  'multiple N' or 'm N'    - Show N random recommendations")
    print("  'id IMAGE_ID'            - Show recommendations for specific ID")
    print("  'index N'                - Show recommendations for index N")
    print("  'compare N'              - Compare N queries side-by-side")
    print("  'list'                   - List first 20 image IDs")
    print("  'quit' or 'q'            - Exit")
    print("=" * 80)
    
    while True:
        try:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif cmd in ['random', 'r']:
                idx = random.randint(0, len(data['product_ids']) - 1)
                print(f"\n📊 Random query index: {idx}")
                print_text_recommendations(data, idx, k=10)
                visualize_recommendations(data, idx, k=10)
            
            elif cmd.startswith('multiple') or cmd.startswith('m'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 5
                visualize_multiple_queries(data, n_queries=n, k=10)
            
            elif cmd.startswith('id'):
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2:
                    print("❌ Usage: id IMAGE_ID")
                    continue
                image_id = parts[1].strip()
                visualize_by_image_id(data, image_id, k=10)
            
            elif cmd.startswith('index'):
                parts = cmd.split()
                if len(parts) < 2:
                    print("❌ Usage: index N")
                    continue
                idx = int(parts[1])
                if 0 <= idx < len(data['product_ids']):
                    print_text_recommendations(data, idx, k=10)
                    visualize_recommendations(data, idx, k=10)
                else:
                    print(f"❌ Index out of range: 0-{len(data['product_ids'])-1}")
            
            elif cmd.startswith('compare'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 3
                indices = random.sample(range(len(data['product_ids'])), n)
                create_comparison_grid(data, indices, k=5)
            
            elif cmd == 'list':
                print("\nFirst 20 image IDs:")
                for i, img_id in enumerate(data['product_ids'][:20], 1):
                    print(f"  {i:2d}. {img_id}")
            
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# ==================== Main ====================
def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("FASHION RECOMMENDATION VISUALIZATION")
    print("=" * 80)
    
    # Load pre-trained system
    data = load_pretrained_system()
    if data is None:
        return
    
    print("\n" + "=" * 80)
    print("QUICK START OPTIONS")
    print("=" * 80)
    print("\n1. Generate 5 random recommendation visualizations")
    print("2. Interactive mode (explore recommendations)")
    print("3. Specific image ID")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == '1':
        print("\n🎨 Generating 5 random recommendations...")
        saved_files = visualize_multiple_queries(data, n_queries=5, k=10)
        print(f"\n✅ Generated {len(saved_files)} visualizations!")
        print(f"📁 Output directory: {Config.OUTPUT_DIR}")
        
    elif choice == '2':
        interactive_mode(data)
        
    elif choice == '3':
        image_id = input("Enter image ID: ").strip()
        visualize_by_image_id(data, image_id, k=10)
        
    elif choice == '4':
        print("👋 Goodbye!")
        return
    
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_mode(data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
