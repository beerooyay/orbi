#!/usr/bin/env python3
"""
arabic wordnet parser for orbihex
extract arabic words and build semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
import re
from collections import defaultdict

class arabic_wordnet_parser:
    def __init__(self, data_file='omw-data/wns/arb/wn-data-arb.tab'):
        self.data_file = data_file
        self.words = set()
        
    def parse_arabic_wordnet(self):
        """parse arabic wordnet data"""
        if not os.path.exists(self.data_file):
            print(f"arabic wordnet not found: {self.data_file}")
            return []
        
        print(f"parsing arabic wordnet from {self.data_file}...")
        
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
                
                # clean word: keep arabic letters, remove diacritics and symbols
                # allow arabic unicode range and basic latin
                clean_word = re.sub(r'[^\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff\w]', '', lemma.lower())
                
                # filter for reasonable length (2-15 chars for arabic)
                if 2 <= len(clean_word) <= 15 and clean_word:
                    words.add(clean_word)
                
                if line_num % 5000 == 0:
                    print(f"  processed {line_num:,} lines, found {len(words)} words")
        
        self.words = sorted(list(words))
        print(f"total arabic words: {len(self.words)}")
        return self.words
    
    def save_arabic_words(self, filename='dataset/ar/words.txt'):
        """save arabic word list"""
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} arabic words to {filename}")

class arabic_lexicon:
    def __init__(self, words, output_file='lexicon/ar/ar.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_arabic_lexicon(self):
        """build arabic lexicon"""
        print(f"building arabic lexicon for {len(self.words)} words...")
        
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
            if processed % 2000 == 0:
                print(f"processed {processed}/{len(self.words)} words ({processed/len(self.words)*100:.1f}%)")
        
        print(f"arabic lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save arabic lexicon"""
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize(self.output_file)
        print(f"saved arabic lexicon to {self.output_file} ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze arabic lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== arabic lexicon analysis ===")
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
    """main arabic wordnet processing"""
    print("=== arabic wordnet orbihex processing ===")
    
    # create directories
    os.makedirs('dataset/ar', exist_ok=True)
    os.makedirs('lexicon/ar', exist_ok=True)
    
    # parse arabic wordnet
    parser = arabic_wordnet_parser()
    arabic_words = parser.parse_arabic_wordnet()
    parser.save_arabic_words()
    
    # build arabic lexicon
    builder = arabic_lexicon(arabic_words)
    arabic_lexicon_data = builder.build_arabic_lexicon()
    
    # analyze
    analysis = builder.analyze_lexicon()
    
    # save
    compression_ratio = builder.save_lexicon()
    
    print(f"\n=== final arabic results ===")
    print(f"lexicon size: {len(arabic_lexicon_data)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return arabic_lexicon_data, analysis

if __name__ == "__main__":
    arabic_lexicon, analysis = main()
