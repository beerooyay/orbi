#!/usr/bin/env python3
"""
italian wordnet parser for orbihex
extract italian words and build semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
from collections import defaultdict

class italian_wordnet_parser:
    def __init__(self, data_file='omw-data/wns/ita/wn-data-ita.tab'):
        self.data_file = data_file
        self.words = set()
        
    def parse_italian_wordnet(self):
        """parse italian wordnet data"""
        if not os.path.exists(self.data_file):
            print(f"italian wordnet not found: {self.data_file}")
            return []
        
        print(f"parsing italian wordnet from {self.data_file}...")
        
        words = set()
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                # format: offset-pos\ttype\tlemma
                lemma = parts[2].strip()
                
                # clean word: remove spaces, special chars, keep italian letters
                clean_word = ''.join(c for c in lemma.lower() if c.isalpha())
                
                if 2 <= len(clean_word) <= 20:
                    words.add(clean_word)
                
                if line_num % 5000 == 0:
                    print(f"  processed {line_num:,} lines, found {len(words)} words")
        
        self.words = sorted(list(words))
        print(f"total italian words: {len(self.words)}")
        return self.words
    
    def save_italian_words(self, filename='dataset/it/words.txt'):
        """save italian word list"""
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} italian words to {filename}")

class italian_lexicon:
    def __init__(self, words, output_file='lexicon/it/it.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_italian_lexicon(self):
        """build italian lexicon"""
        print(f"building italian lexicon for {len(self.words)} words...")
        
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
            if processed % 3000 == 0:
                print(f"processed {processed}/{len(self.words)} words ({processed/len(self.words)*100:.1f}%)")
        
        print(f"italian lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save italian lexicon"""
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize(self.output_file)
        print(f"saved italian lexicon to {self.output_file} ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze italian lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== italian lexicon analysis ===")
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
        
        return {
            'total_words': len(self.lexicon),
            'mass_range': (min(masses), max(masses)),
            'avg_mass': np.mean(masses),
            'length_mass_correlation': length_mass_corr
        }

def main():
    """main italian wordnet processing"""
    print("=== italian wordnet orbihex processing ===")
    
    # create directories
    os.makedirs('dataset/it', exist_ok=True)
    os.makedirs('lexicon/it', exist_ok=True)
    
    # parse italian wordnet
    parser = italian_wordnet_parser()
    italian_words = parser.parse_italian_wordnet()
    parser.save_italian_words()
    
    # build italian lexicon
    builder = italian_lexicon(italian_words)
    italian_lexicon_data = builder.build_italian_lexicon()
    
    # analyze
    analysis = builder.analyze_lexicon()
    
    # save
    compression_ratio = builder.save_lexicon()
    
    print(f"\n=== final italian results ===")
    print(f"lexicon size: {len(italian_lexicon_data)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return italian_lexicon_data, analysis

if __name__ == "__main__":
    italian_lexicon, analysis = main()
