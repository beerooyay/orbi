#!/usr/bin/env python3
"""
parse wordnet database to build complete semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
from collections import defaultdict

class wordnet_parser:
    def __init__(self, dict_path='dataset/en/dict'):
        self.dict_path = dict_path
        self.words = set()
        
    def parse_index_file(self, filename):
        """parse wordnet index file to extract words"""
        words = set()
        file_path = os.path.join(self.dict_path, filename)
        
        if not os.path.exists(file_path):
            print(f"file not found: {file_path}")
            return words
            
        print(f"parsing {filename}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('  '):
                    # extract word (first token before space)
                    word = line.split()[0].lower()
                    # clean word: remove numbers, underscores, etc.
                    word = ''.join(c for c in word if c.isalpha())
                    if 2 <= len(word) <= 20:
                        words.add(word)
        
        print(f"found {len(words)} words in {filename}")
        return words
    
    def extract_all_words(self):
        """extract words from all index files"""
        index_files = ['index.noun', 'index.verb', 'index.adj', 'index.adv']
        
        all_words = set()
        for filename in index_files:
            words = self.parse_index_file(filename)
            all_words.update(words)
        
        self.words = sorted(list(all_words))
        print(f"total unique words: {len(self.words)}")
        return self.words
    
    def save_word_list(self, filename='dataset/en/words.txt'):
        """save word list to file"""
        with open(filename, 'w') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} words to {filename}")

class english_lexicon:
    def __init__(self, words, output_file='lexicon/en/en.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_lexicon(self):
        """build complete lexicon from word list"""
        print(f"building lexicon for {len(self.words)} words...")
        
        # phase basis for nibble mapping
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        basis = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        processed = 0
        for word in self.words:
            nibbles = self.word_to_nibbles(word)
            if not nibbles:
                continue
            
            # compute semantic trajectory
            h = np.zeros(2)
            for t, n in enumerate(nibbles):
                dh = basis[n]
                h = (h * t + dh) / (t + 1)
            
            # compute semantic metrics
            com = h
            mass = np.linalg.norm(com)
            phase = 2 * np.pi * np.mean(nibbles) / 16
            
            self.lexicon[word] = {
                'com': com.tolist(),
                'mass': float(mass),
                'phase': float(phase),
                'nibbles': nibbles,
                'length': len(word)
            }
            
            processed += 1
            if processed % 1000 == 0:
                print(f"processed {processed}/{len(self.words)} words ({processed/len(self.words)*100:.1f}%)")
        
        print(f"lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save lexicon to file"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize(self.output_file)
        print(f"saved lexicon to {self.output_file} ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze the built lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== full wordnet lexicon analysis ===")
        print(f"total words: {len(self.lexicon)}")
        print(f"mass range: {min(masses):.3f} - {max(masses):.3f}")
        print(f"avg mass: {np.mean(masses):.3f}")
        print(f"phase coverage: {min(phases):.3f} - {max(phases):.3f}")
        print(f"length range: {min(lengths)} - {max(lengths)} chars")
        
        # mass distribution
        mass_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nmass distribution:")
        for p in mass_percentiles:
            value = np.percentile(masses, p)
            print(f"  {p}th percentile: {value:.3f}")
        
        # length vs mass correlation
        length_mass_corr = np.corrcoef(lengths, masses)[0, 1]
        print(f"\nlength vs mass correlation: {length_mass_corr:.3f}")
        
        # find examples
        sorted_by_mass = sorted(self.lexicon.items(), key=lambda x: x[1]['mass'])
        print(f"\nextreme mass examples:")
        print(f"lowest mass: {sorted_by_mass[0][0]} ({sorted_by_mass[0][1]['mass']:.3f})")
        print(f"highest mass: {sorted_by_mass[-1][0]} ({sorted_by_mass[-1][1]['mass']:.3f})")
        
        return {
            'total_words': len(self.lexicon),
            'mass_range': (min(masses), max(masses)),
            'avg_mass': np.mean(masses),
            'length_mass_correlation': length_mass_corr
        }

def main():
    """main entry point"""
    print("=== building full wordnet lexicon ===")
    
    # parse wordnet
    parser = wordnet_parser()
    words = parser.extract_all_words()
    parser.save_word_list('dataset/en/words.txt')
    
    # build lexicon
    lexicon_builder = english_lexicon(words)
    lexicon = lexicon_builder.build_lexicon()
    
    # analyze
    analysis = lexicon_builder.analyze_lexicon()
    
    # save
    compression_ratio = lexicon_builder.save_lexicon()
    
    print(f"\n=== final results ===")
    print(f"lexicon size: {len(lexicon)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return lexicon, analysis

if __name__ == "__main__":
    lexicon, analysis = main()
