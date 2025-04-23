# policy_heatmap.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def load_policy_data():
    """Load policy evaluation data from analysis report"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(current_dir, "财政补贴政策_分析报告.txt")
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sample data structure (replace with actual data parsing)
        scores = {
            "评估维度": ["资金规模", "时限要求", "追责条款", "跨部门协同", "量化目标"],
            "财政补贴政策": [4, 3, 2, 1, 3],
            "产业发展规划": [5, 5, 4, 3, 5],
            "科技创新政策": [3, 4, 3, 2, 4],
            "人才引进政策": [2, 3, 4, 5, 3]
        }
        
        return pd.DataFrame(scores).set_index("评估维度")
    
    except Exception as e:
        print(f"Error loading policy data: {str(e)}")
        return None

def set_chinese_font():
    """Configure Chinese font support"""
    try:
        # Try using Microsoft YaHei (Windows)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        try:
            # Fallback to SimHei (Windows)
            plt.rcParams['font.sans-serif'] = ['SimHei']
        except:
            try:
                # Fallback to Arial Unicode MS (Mac)
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            except:
                print("Warning: Chinese font not found, display may be incorrect")

def generate_heatmap(data):
    """Generate a professional heatmap visualization"""
    # Set Chinese font
    set_chinese_font()
    
    # Set up figure with adjusted size
    plt.figure(figsize=(12, 7))
    
    # Create colorblind-friendly colormap (blue to purple)
    cmap = LinearSegmentedColormap.from_list(
        'policy_cmap', ['#f7fcf5', '#74c476', '#00441b'], N=256)
    
    # Create heatmap using seaborn
    ax = sns.heatmap(
        data,
        cmap=cmap,
        annot=True,
        fmt="d",
        annot_kws={"size": 12, "color": "black"},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '政策强度评分', 'pad': 0.02}
    )
    
    # Customize appearance
    plt.title("政策强度五维评估热力图\n基于结构化分析结果", 
              fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("政策文件", fontsize=12, labelpad=10)
    plt.ylabel("评估维度", fontsize=12, labelpad=10)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add scoring criteria annotation with adjusted position
    plt.figtext(
        0.78,  # Further adjusted x-position (0-1, from left)
        0.65,  # y-position (0-1, from bottom)
        "评分标准：\n"
        "5分：量化指标+跨部门协同+追责条款\n"
        "4分：多措施组合+时限要求\n"
        "3分：具体措施+有限资源",
        ha="left", 
        va="center", 
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    # Adjust layout with more padding
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Adjusted to leave more space
    
    # Save and show
    output_path = os.path.join(os.path.dirname(__file__), "policy_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Load data
    policy_data = load_policy_data()
    
    if policy_data is not None:
        # Generate visualization
        generate_heatmap(policy_data)