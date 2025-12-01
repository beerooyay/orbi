#!/usr/bin/env python3
"""
spanish wordnet parser for orbihex
extract spanish words and build semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
from collections import defaultdict

class spanish_wordnet_parser:
    def __init__(self, data_file='omw-data/wns/mcr/wn-data-spa.tab'):
        self.data_file = data_file
        self.words = set()
        
    def parse_spanish_wordnet(self):
        """parse spanish wordnet data"""
        if not os.path.exists(self.data_file):
            print(f"spanish wordnet not found: {self.data_file}")
            return []
        
        print(f"parsing spanish wordnet from {self.data_file}...")
        
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
                
                # clean word: remove spaces, accents, special chars
                clean_word = ''.join(c for c in lemma.lower() if c.isalpha())
                
                if 2 <= len(clean_word) <= 20:
                    words.add(clean_word)
                
                if line_num % 10000 == 0:
                    print(f"  processed {line_num:,} lines, found {len(words)} words")
        
        self.words = sorted(list(words))
        print(f"total spanish words: {len(self.words)}")
        return self.words
    
    def save_spanish_words(self, filename='dataset/es/words.txt'):
        """save spanish word list"""
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} spanish words to {filename}")

class spanish_lexicon:
    def __init__(self, words, output_file='lexicon/es/es.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_spanish_lexicon(self):
        """build spanish lexicon"""
        print(f"building spanish lexicon for {len(self.words)} words...")
        
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
        
        print(f"spanish lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save spanish lexicon"""
        with open('lexicon/es/es.pkl', 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize('lexicon/es/es.pkl')
        print(f"saved spanish lexicon to lexicon/es/es.pkl ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze spanish lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== spanish lexicon analysis ===")
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

def compare_english_spanish(english_lexicon='lexicon/en/en.pkl', spanish_lexicon='lexicon/es/es.pkl'):
    """compare english and spanish lexicon geometries"""
    print("\n=== english vs spanish comparison ===")
    
    # load both lexicons
    with open(english_lexicon, 'rb') as f:
        english_data = pickle.load(f)
    
    with open(spanish_lexicon, 'rb') as f:
        spanish_data = pickle.load(f)
    
    # extract statistics
    english_masses = [data['mass'] for data in english_data.values()]
    spanish_masses = [data['mass'] for data in spanish_data.values()]
    
    english_phases = [data['phase'] for data in english_data.values()]
    spanish_phases = [data['phase'] for data in spanish_data.values()]
    
    # mass distribution comparison
    print(f"mass distribution comparison:")
    print(f"  english: {len(english_data):,} words, range {min(english_masses):.3f} - {max(english_masses):.3f}")
    print(f"  spanish: {len(spanish_data):,} words, range {min(spanish_masses):.3f} - {max(spanish_masses):.3f}")
    
    # test for bimodal distributions
    english_hist, english_bins = np.histogram(english_masses, bins=50)
    spanish_hist, spanish_bins = np.histogram(spanish_masses, bins=50)
    
    # find peaks
    def find_peaks(hist, bins):
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bins[i], hist[i]))
        return peaks
    
    english_peaks = find_peaks(english_hist, english_bins)
    spanish_peaks = find_peaks(spanish_hist, spanish_bins)
    
    print(f"\nmass distribution peaks:")
    print(f"  english: {len(english_peaks)} peaks")
    for bin_center, count in english_peaks[:3]:
        print(f"    mass {bin_center:.3f}: {count} words")
    
    print(f"  spanish: {len(spanish_peaks)} peaks")
    for bin_center, count in spanish_peaks[:3]:
        print(f"    mass {bin_center:.3f}: {count} words")
    
    # phase distribution comparison
    phase_bins = np.linspace(0, 2*np.pi, 17)
    english_phase_hist, _ = np.histogram(english_phases, bins=phase_bins)
    spanish_phase_hist, _ = np.histogram(spanish_phases, bins=phase_bins)
    
    # significant phase clusters
    english_clusters = [(i, count) for i, count in enumerate(english_phase_hist) if count > 1000]
    spanish_clusters = [(i, count) for i, count in enumerate(spanish_phase_hist) if count > 1000]
    
    print(f"\nphase cluster comparison:")
    print(f"  english: {len(english_clusters)} significant clusters")
    print(f"  spanish: {len(spanish_clusters)} significant clusters")
    
    # similarity assessment
    mass_similarity = 1 - abs(np.mean(english_masses) - np.mean(spanish_masses))
    phase_similarity = 1 - abs(np.mean(english_phases) - np.mean(spanish_phases)) / (2*np.pi)
    
    print(f"\nsimilarity assessment:")
    print(f"  mass distribution similarity: {mass_similarity:.3f}")
    print(f"  phase distribution similarity: {phase_similarity:.3f}")
    
    # universal vs language-specific
    if len(english_peaks) == len(spanish_peaks) and len(english_clusters) == len(spanish_clusters):
        print(f"\n🔥 RESULT: UNIVERSAL GEOMETRY DETECTED!")
        print(f"   Same number of mass peaks ({len(english_peaks)}) and phase clusters ({len(english_clusters)})")
    else:
        print(f"\n🌍 RESULT: LANGUAGE-SPECIFIC SIGNATURES!")
        print(f"   English: {len(english_peaks)} mass peaks, {len(english_clusters)} phase clusters")
        print(f"   Spanish: {len(spanish_peaks)} mass peaks, {len(spanish_clusters)} phase clusters")
    
    return {
        'english_peaks': len(english_peaks),
        'spanish_peaks': len(spanish_peaks),
        'english_clusters': len(english_clusters),
        'spanish_clusters': len(spanish_clusters),
        'mass_similarity': mass_similarity,
        'phase_similarity': phase_similarity
    }

def main():
    """main spanish wordnet processing"""
    print("=== spanish wordnet orbihex processing ===")
    
    # parse spanish wordnet
    parser = spanish_wordnet_parser()
    spanish_words = parser.parse_spanish_wordnet()
    parser.save_spanish_words()
    
    # build spanish lexicon
    builder = spanish_lexicon(spanish_words)
    spanish_lexicon_data = builder.build_spanish_lexicon()
    
    # analyze
    analysis = builder.analyze_lexicon()
    
    # save
    compression_ratio = builder.save_lexicon()
    
    # compare with english
    comparison = compare_english_spanish()
    
    print(f"\n=== final spanish results ===")
    print(f"lexicon size: {len(spanish_lexicon_data)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return spanish_lexicon_data, analysis, comparison

if __name__ == "__main__":
    spanish_lexicon, analysis, comparison = main()
