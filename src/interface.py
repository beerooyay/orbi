#!/usr/bin/env python3
"""
orbihex analysis interface
analyze and assess semantic structure at scale
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.orbihex import orbihex

class orbihex_analyzer:
    def __init__(self, lexicon_file='lexicon/en/en.pkl'):
        self.system = orbihex(lexicon_file=lexicon_file)
        print(f"orbihex analyzer ready: {len(self.system.words)} words")
    
    def semantic_structure_analysis(self):
        """analyze semantic structure preservation"""
        print("\n=== semantic structure analysis ===")
        
        # semantic search quality
        test_words = ['happy', 'sad', 'angry', 'love', 'hate', 'life', 'death', 'time', 'space', 'mind']
        
        print("semantic neighbors:")
        for word in test_words:
            if word in self.system.lexicon:
                neighbors = self.system.search(word, n_neighbors=3)
                print(f"  {word}: {[f'{n}({d:.3f})' for n, d in neighbors]}")
        
        # semantic mass distribution
        masses = self.system.mass_vector
        print(f"\nsemantic mass distribution:")
        print(f"  mean: {np.mean(masses):.3f}")
        print(f"  std: {np.std(masses):.3f}")
        print(f"  range: {np.min(masses):.3f} - {np.max(masses):.3f}")
        
        # mass vs length correlation
        lengths = np.array([self.system.lexicon[w]['length'] for w in self.system.words])
        corr = np.corrcoef(lengths, masses)[0, 1]
        print(f"  length-mass correlation: {corr:.3f} (should be near 0)")
        
        return {
            'mass_stats': {
                'mean': np.mean(masses),
                'std': np.std(masses),
                'range': (np.min(masses), np.max(masses))
            },
            'length_mass_correlation': corr
        }
    
    def clustering_analysis(self, n_clusters=8):
        """analyze semantic clustering"""
        print(f"\n=== clustering analysis ({n_clusters} clusters) ===")
        
        clusters = self.system.cluster(n_clusters=n_clusters)
        cluster_analysis = self.system.analyze_clusters(clusters)
        
        print("cluster characteristics:")
        for cluster_id, analysis in cluster_analysis.items():
            print(f"  cluster {cluster_id}: {analysis['size']} words")
            print(f"    avg mass: {analysis['avg_mass']:.3f} ± {analysis['mass_std']:.3f}")
            print(f"    sample: {analysis['sample_words'][:3]}")
        
        # cluster mass separation
        cluster_masses = [analysis['avg_mass'] for analysis in cluster_analysis.values()]
        mass_separation = np.std(cluster_masses)
        print(f"\ncluster mass separation: {mass_separation:.3f}")
        
        return cluster_analysis
    
    def dimensionality_analysis(self):
        """analyze dimensionality and compression"""
        print("\n=== dimensionality analysis ===")
        
        # PCA variance explained
        pca = self.system.project_pca(n_components=2)
        explained_variance = self.system.pca.explained_variance_ratio_
        
        print(f"PCA variance explained:")
        print(f"  PC1: {explained_variance[0]:.3f}")
        print(f"  PC2: {explained_variance[1]:.3f}")
        print(f"  total: {sum(explained_variance):.3f}")
        
        # compression analysis
        stats = self.system.stats()
        print(f"\ncompression analysis:")
        print(f"  words: {stats['total_words']:,}")
        print(f"  file size: {os.path.getsize(self.system.lexicon_file)/1e6:.1f} MB")
        print(f"  compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"  traditional would be: {stats['total_words'] * 300 * 4 / 1e6:.1f} MB")
        
        return {
            'pca_variance': explained_variance,
            'compression': stats['compression_ratio']
        }
    
    def machine_learning_readiness(self):
        """assess machine learning readiness"""
        print("\n=== machine learning readiness ===")
        
        # feature preprocessing
        features = self.system.preprocess_features()
        print(f"features extracted: {features.shape}")
        print(f"feature names: {self.system.feature_names}")
        
        # label creation
        labels = self.system.create_labels(method='mass_quantiles', n_classes=3)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"\nlabel distribution:")
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            print(f"  class {label}: {count:,} words ({count/len(labels)*100:.1f}%)")
        
        # data splitting
        data = self.system.split_data(test_size=0.2)
        print(f"\ndata split:")
        print(f"  train: {data['train']['X'].shape[0]:,} samples")
        print(f"  test: {data['test']['X'].shape[0]:,} samples")
        
        # pytorch export
        torch_data = self.system.export_torch(data)
        if torch_data:
            print(f"\npytorch datasets created:")
            print(f"  train batches: {len(torch_data['train_loader'])}")
            print(f"  test batches: {len(torch_data['test_loader'])}")
        
        return {
            'features': features.shape,
            'labels': dict(zip(unique_labels, counts)),
            'data_split': {
                'train': data['train']['X'].shape[0],
                'test': data['test']['X'].shape[0]
            }
        }
    
    def semantic_mass_analysis(self):
        """deep semantic mass analysis"""
        print("\n=== semantic mass analysis ===")
        
        masses = self.system.mass_vector
        
        # mass percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("mass percentiles:")
        for p in percentiles:
            value = np.percentile(masses, p)
            print(f"  {p}th: {value:.3f}")
        
        # extreme mass words
        high_mass = self.system.filter_by_mass(min_mass=np.percentile(masses, 99), limit=10)
        low_mass = self.system.filter_by_mass(max_mass=np.percentile(masses, 1), limit=10)
        
        print(f"\nhighest mass words (top 1%):")
        for word, mass in high_mass:
            print(f"  {word}: {mass:.3f}")
        
        print(f"\nlowest mass words (bottom 1%):")
        for word, mass in low_mass:
            print(f"  {word}: {mass:.3f}")
        
        # mass categories
        low_threshold = np.percentile(masses, 33)
        high_threshold = np.percentile(masses, 67)
        
        low_mass_words = [w for w in self.system.words if self.system.lexicon[w]['mass'] < low_threshold]
        high_mass_words = [w for w in self.system.words if self.system.lexicon[w]['mass'] > high_threshold]
        
        print(f"\nmass categories:")
        print(f"  low mass (< {low_threshold:.3f}): {len(low_mass_words):,} words")
        print(f"  medium mass: {len(self.system.words) - len(low_mass_words) - len(high_mass_words):,} words")
        print(f"  high mass (> {high_threshold:.3f}): {len(high_mass_words):,} words")
        
        return {
            'percentiles': {p: np.percentile(masses, p) for p in percentiles},
            'extreme_words': {
                'high': [(w, m) for w, m in high_mass],
                'low': [(w, m) for w, m in low_mass]
            },
            'categories': {
                'low': len(low_mass_words),
                'medium': len(self.system.words) - len(low_mass_words) - len(high_mass_words),
                'high': len(high_mass_words)
            }
        }
    
    def phase_analysis(self):
        """analyze phase relationships"""
        print("\n=== phase analysis ===")
        
        phases = self.system.phase_vector
        
        print(f"phase statistics:")
        print(f"  range: {np.min(phases):.3f} - {np.max(phases):.3f}")
        print(f"  mean: {np.mean(phases):.3f}")
        print(f"  coverage: {(np.max(phases) - np.min(phases))/(2*np.pi)*100:.1f}% of full circle")
        
        # phase vs mass correlation
        masses = self.system.mass_vector
        phase_mass_corr = np.corrcoef(phases, masses)[0, 1]
        print(f"  phase-mass correlation: {phase_mass_corr:.3f}")
        
        # phase distribution
        phase_bins = np.linspace(0, 2*np.pi, 9)  # 8 bins
        phase_hist, _ = np.histogram(phases, bins=phase_bins)
        
        print(f"\nphase distribution (8 bins):")
        for i, count in enumerate(phase_hist):
            print(f"  bin {i+1}: {count:,} words ({count/len(phases)*100:.1f}%)")
        
        return {
            'phase_stats': {
                'range': (np.min(phases), np.max(phases)),
                'mean': np.mean(phases),
                'coverage': (np.max(phases) - np.min(phases))/(2*np.pi)
            },
            'phase_mass_correlation': phase_mass_corr,
            'phase_distribution': phase_hist.tolist()
        }
    
    def comprehensive_assessment(self):
        """run comprehensive assessment"""
        print("=== comprehensive orbihex assessment ===")
        
        # run all analyses
        semantic_results = self.semantic_structure_analysis()
        clustering_results = self.clustering_analysis()
        dimensionality_results = self.dimensionality_analysis()
        ml_results = self.machine_learning_readiness()
        mass_results = self.semantic_mass_analysis()
        phase_results = self.phase_analysis()
        
        # summary
        print(f"\n=== assessment summary ===")
        print(f"semantic structure: {'preserved' if semantic_results['length_mass_correlation'] < 0.1 else 'degraded'}")
        print(f"clustering quality: {'good' if len(clustering_results) >= 6 else 'limited'}")
        print(f"compression efficiency: {'excellent' if dimensionality_results['compression'] > 10 else 'moderate'}")
        print(f"ml readiness: {'ready' if ml_results['features'][0] > 100000 else 'limited'}")
        print(f"mass distribution: {'healthy' if 0.2 < mass_results['percentiles'][50] < 0.8 else 'skewed'}")
        print(f"phase coverage: {'good' if phase_results['phase_stats']['coverage'] > 0.8 else 'limited'}")
        
        return {
            'semantic': semantic_results,
            'clustering': clustering_results,
            'dimensionality': dimensionality_results,
            'ml_readiness': ml_results,
            'mass_analysis': mass_results,
            'phase_analysis': phase_results
        }

def main():
    """main analysis interface"""
    analyzer = orbihex_analyzer()
    results = analyzer.comprehensive_assessment()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()
