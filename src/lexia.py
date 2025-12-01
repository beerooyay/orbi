#!/usr/bin/env python3
"""
orbihex + leif integration
lexical mask retrieval system
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.orbihex import orbihex

class lexia:
    def __init__(self, lexicon_file='lexicon/en/en.pkl'):
        self.system = orbihex(lexicon_file=lexicon_file)
        print(f"lexia retrieval ready: {len(self.system.words)} words")
    
    def lexical_mask_search(self, query_word, mask_type='all'):
        """search using lexical mask patterns (fast version)"""
        if query_word not in self.system.lexicon:
            return []
        
        query_data = self.system.lexicon[query_word]
        query_mass = query_data['mass']
        query_phase = query_data['phase']
        query_com = np.array(query_data['com'])
        
        results = []
        
        # sample candidates for speed
        sample_size = min(1000, len(self.system.words))
        sample_indices = np.random.choice(len(self.system.words), sample_size, replace=False)
        
        for idx in sample_indices:
            word = self.system.words[idx]
            if word == query_word:
                continue
            
            data = self.system.lexicon[word]
            com = np.array(data['com'])
            dist = np.linalg.norm(query_com - com)
            
            if dist > 0.2:  # too far
                continue
            
            # lexical mask conditions
            if mask_type == 'all' or mask_type == 'same_sender':
                mass_diff = abs(query_mass - data['mass'])
                if mass_diff < 0.1:  # similar mass
                    results.append((word, dist, mass_diff))
            
            elif mask_type == 'direct_address':
                mass_diff = abs(query_mass - data['mass'])
                if 0.3 < mass_diff < 0.5:  # complementary mass
                    results.append((word, dist, mass_diff))
            
            elif mask_type == 'temporal_neighbors':
                phase_diff = abs(query_phase - data['phase'])
                if phase_diff < 0.5:  # similar phase
                    results.append((word, dist, phase_diff))
        
        # sort by distance
        results.sort(key=lambda x: x[1])
        return results[:10]
    
    def enhanced_retrieval_test(self):
        """test enhanced retrieval with lexical masks"""
        print("\n=== enhanced lexical retrieval test ===")
        
        test_words = ['happy', 'sad', 'love', 'hate', 'life', 'death', 'time', 'space']
        
        for word in test_words:
            if word not in self.system.lexicon:
                continue
            
            print(f"\n{word}:")
            
            # standard search
            standard_neighbors = self.system.search(word, n_neighbors=3)
            print(f"  standard: {[n for n, d in standard_neighbors]}")
            
            # lexical mask search
            masked_neighbors = self.lexical_mask_search(word, mask_type='all')
            print(f"  lexical: {[n for n, d, diff in masked_neighbors[:3]]}")
    
    def spanish_compatibility_test(self):
        """prepare for spanish wordnet comparison"""
        print("\n=== spanish compatibility analysis ===")
        
        # identify universal vs language-specific patterns
        universal_patterns = {
            'mass_distribution': self.system.mass_vector,
            'phase_distribution': self.system.phase_vector,
            'com_distribution': self.system.com_matrix
        }
        
        # save patterns for comparison
        np.save('lexicon/patterns.npy', universal_patterns)
        
        print(f"saved universal patterns for spanish comparison:")
        print(f"  mass range: {np.min(self.system.mass_vector):.3f} - {np.max(self.system.mass_vector):.3f}")
        print(f"  phase range: {np.min(self.system.phase_vector):.3f} - {np.max(self.system.phase_vector):.3f}")
        print(f"  com range: [{np.min(self.system.com_matrix, axis=0)}, {np.max(self.system.com_matrix, axis=0)}]")
        
        return universal_patterns

def main():
    """main lexia retrieval system"""
    retrieval = lexia()
    
    # run enhanced tests
    retrieval.enhanced_retrieval_test()
    patterns = retrieval.spanish_compatibility_test()
    
    return retrieval, patterns

if __name__ == "__main__":
    retrieval, patterns = main()
