#!/usr/bin/env python3
"""
japanese wordnet parser for orbihex
extract japanese words and build semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
import re
from collections import defaultdict

class japanese_wordnet_parser:
    def __init__(self, data_file='omw-data/wns/jpn/wn-data-jpn.tab'):
        self.data_file = data_file
        self.words = set()
        
    def parse_japanese_wordnet(self):
        """parse japanese wordnet data"""
        if not os.path.exists(self.data_file):
            print(f"japanese wordnet not found: {self.data_file}")
            return []
        
        print(f"parsing japanese wordnet from {self.data_file}...")
        
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
                
                # clean word: keep hiragana, katakana, kanji, remove symbols
                # allow japanese characters and basic latin letters
                clean_word = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\w]', '', lemma.lower())
                
                # filter for reasonable length (1-10 chars for japanese)
                if 1 <= len(clean_word) <= 10 and clean_word:
                    words.add(clean_word)
                
                if line_num % 50000 == 0:
                    print(f"  processed {line_num:,} lines, found {len(words)} words")
        
        self.words = sorted(list(words))
        print(f"total japanese words: {len(self.words)}")
        return self.words
    
    def save_japanese_words(self, filename='dataset/ja/words.txt'):
        """save japanese word list"""
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} japanese words to {filename}")

class japanese_lexicon:
    def __init__(self, words, output_file='lexicon/ja/ja.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_japanese_lexicon(self):
        """build japanese lexicon"""
        print(f"building japanese lexicon for {len(self.words)} words...")
        
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
            if processed % 5000 == 0:
                print(f"processed {processed}/{len(self.words)} words ({processed/len(self.words)*100:.1f}%)")
        
        print(f"japanese lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save japanese lexicon"""
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize(self.output_file)
        print(f"saved japanese lexicon to {self.output_file} ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze japanese lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== japanese lexicon analysis ===")
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

def compare_six_languages(english_lexicon='lexicon/en/en.pkl', 
                         spanish_lexicon='lexicon/es/es.pkl', 
                         french_lexicon='lexicon/fr/fr.pkl',
                         norwegian_lexicon='lexicon/no/no.pkl',
                         finnish_lexicon='lexicon/fi/fi.pkl',
                         japanese_lexicon='lexicon/ja/ja.pkl'):
    """compare six languages lexicon geometries"""
    print("\n=== six language comparison ===")
    
    # load all lexicons
    with open(english_lexicon, 'rb') as f:
        english_data = pickle.load(f)
    
    with open(spanish_lexicon, 'rb') as f:
        spanish_data = pickle.load(f)
    
    with open(french_lexicon, 'rb') as f:
        french_data = pickle.load(f)
    
    with open(norwegian_lexicon, 'rb') as f:
        norwegian_data = pickle.load(f)
    
    with open(finnish_lexicon, 'rb') as f:
        finnish_data = pickle.load(f)
    
    with open(japanese_lexicon, 'rb') as f:
        japanese_data = pickle.load(f)
    
    # extract statistics
    english_masses = [data['mass'] for data in english_data.values()]
    spanish_masses = [data['mass'] for data in spanish_data.values()]
    french_masses = [data['mass'] for data in french_data.values()]
    norwegian_masses = [data['mass'] for data in norwegian_data.values()]
    finnish_masses = [data['mass'] for data in finnish_data.values()]
    japanese_masses = [data['mass'] for data in japanese_data.values()]
    
    english_phases = [data['phase'] for data in english_data.values()]
    spanish_phases = [data['phase'] for data in spanish_data.values()]
    french_phases = [data['phase'] for data in french_data.values()]
    norwegian_phases = [data['phase'] for data in norwegian_data.values()]
    finnish_phases = [data['phase'] for data in finnish_data.values()]
    japanese_phases = [data['phase'] for data in japanese_data.values()]
    
    # mass distribution comparison
    print(f"mass distribution comparison:")
    print(f"  english: {len(english_data):,} words, range {min(english_masses):.3f} - {max(english_masses):.3f}")
    print(f"  spanish: {len(spanish_data):,} words, range {min(spanish_masses):.3f} - {max(spanish_masses):.3f}")
    print(f"  french: {len(french_data):,} words, range {min(french_masses):.3f} - {max(french_masses):.3f}")
    print(f"  norwegian: {len(norwegian_data):,} words, range {min(norwegian_masses):.3f} - {max(norwegian_masses):.3f}")
    print(f"  finnish: {len(finnish_data):,} words, range {min(finnish_masses):.3f} - {max(finnish_masses):.3f}")
    print(f"  japanese: {len(japanese_data):,} words, range {min(japanese_masses):.3f} - {max(japanese_masses):.3f}")
    
    # test for mass distribution peaks
    def count_peaks(masses, bins=50):
        hist, bins = np.histogram(masses, bins=bins)
        peaks = 0
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 1000:
                peaks += 1
        return peaks
    
    english_peaks = count_peaks(english_masses)
    spanish_peaks = count_peaks(spanish_masses)
    french_peaks = count_peaks(french_masses)
    norwegian_peaks = count_peaks(norwegian_masses)
    finnish_peaks = count_peaks(finnish_masses)
    japanese_peaks = count_peaks(japanese_masses)
    
    print(f"\nmass distribution peaks:")
    print(f"  english: {english_peaks} peaks")
    print(f"  spanish: {spanish_peaks} peaks")
    print(f"  french: {french_peaks} peaks")
    print(f"  norwegian: {norwegian_peaks} peaks")
    print(f"  finnish: {finnish_peaks} peaks")
    print(f"  japanese: {japanese_peaks} peaks")
    
    # phase distribution comparison
    phase_bins = np.linspace(0, 2*np.pi, 17)
    english_phase_hist, _ = np.histogram(english_phases, bins=phase_bins)
    spanish_phase_hist, _ = np.histogram(spanish_phases, bins=phase_bins)
    french_phase_hist, _ = np.histogram(french_phases, bins=phase_bins)
    norwegian_phase_hist, _ = np.histogram(norwegian_phases, bins=phase_bins)
    finnish_phase_hist, _ = np.histogram(finnish_phases, bins=phase_bins)
    japanese_phase_hist, _ = np.histogram(japanese_phases, bins=phase_bins)
    
    # significant phase clusters
    def count_clusters(hist):
        return sum(1 for count in hist if count > 1000)
    
    english_clusters = count_clusters(english_phase_hist)
    spanish_clusters = count_clusters(spanish_phase_hist)
    french_clusters = count_clusters(french_phase_hist)
    norwegian_clusters = count_clusters(norwegian_phase_hist)
    finnish_clusters = count_clusters(finnish_phase_hist)
    japanese_clusters = count_clusters(japanese_phase_hist)
    
    print(f"\nphase cluster comparison:")
    print(f"  english: {english_clusters} significant clusters")
    print(f"  spanish: {spanish_clusters} significant clusters")
    print(f"  french: {french_clusters} significant clusters")
    print(f"  norwegian: {norwegian_clusters} significant clusters")
    print(f"  finnish: {finnish_clusters} significant clusters")
    print(f"  japanese: {japanese_clusters} significant clusters")
    
    # universal vs language-specific
    all_clusters = [english_clusters, spanish_clusters, french_clusters, norwegian_clusters, finnish_clusters, japanese_clusters]
    if len(set(all_clusters)) == 1:
        print(f"\nresult: universal phase topology confirmed!")
        print(f"   all {len(all_clusters)} languages show {english_clusters} phase clusters")
    else:
        print(f"\nresult: language-specific phase signatures detected")
        print(f"   cluster distribution: {dict(zip(['en','es','fr','no','fi','ja'], all_clusters))}")
    
    return {
        'peaks': {'en': english_peaks, 'es': spanish_peaks, 'fr': french_peaks, 'no': norwegian_peaks, 'fi': finnish_peaks, 'ja': japanese_peaks},
        'clusters': {'en': english_clusters, 'es': spanish_clusters, 'fr': french_clusters, 'no': norwegian_clusters, 'fi': finnish_clusters, 'ja': japanese_clusters},
        'word_counts': {'en': len(english_data), 'es': len(spanish_data), 'fr': len(french_data), 'no': len(norwegian_data), 'fi': len(finnish_data), 'ja': len(japanese_data)}
    }

def main():
    """main japanese wordnet processing"""
    print("=== japanese wordnet orbihex processing ===")
    
    # create directories
    os.makedirs('dataset/ja', exist_ok=True)
    os.makedirs('lexicon/ja', exist_ok=True)
    
    # parse japanese wordnet
    parser = japanese_wordnet_parser()
    japanese_words = parser.parse_japanese_wordnet()
    parser.save_japanese_words()
    
    # build japanese lexicon
    builder = japanese_lexicon(japanese_words)
    japanese_lexicon_data = builder.build_japanese_lexicon()
    
    # analyze
    analysis = builder.analyze_lexicon()
    
    # save
    compression_ratio = builder.save_lexicon()
    
    # compare with other languages
    comparison = compare_six_languages()
    
    print(f"\n=== final japanese results ===")
    print(f"lexicon size: {len(japanese_lexicon_data)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return japanese_lexicon_data, analysis, comparison

if __name__ == "__main__":
    japanese_lexicon, analysis, comparison = main()
