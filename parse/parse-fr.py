#!/usr/bin/env python3
"""
french wordnet parser for orbihex
extract french words and build semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
from collections import defaultdict

class french_wordnet_parser:
    def __init__(self, data_file='omw-data/wns/fra/wn-data-fra.tab'):
        self.data_file = data_file
        self.words = set()
        
    def parse_french_wordnet(self):
        """parse french wordnet data"""
        if not os.path.exists(self.data_file):
            print(f"french wordnet not found: {self.data_file}")
            return []
        
        print(f"parsing french wordnet from {self.data_file}...")
        
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
        print(f"total french words: {len(self.words)}")
        return self.words
    
    def save_french_words(self, filename='dataset/fr/words.txt'):
        """save french word list"""
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} french words to {filename}")

class french_lexicon:
    def __init__(self, words, output_file='lexicon/fr/fr.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_french_lexicon(self):
        """build french lexicon"""
        print(f"building french lexicon for {len(self.words)} words...")
        
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
        
        print(f"french lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save french lexicon"""
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize(self.output_file)
        print(f"saved french lexicon to {self.output_file} ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze french lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== french lexicon analysis ===")
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

def compare_three_languages(english_lexicon='lexicon/en/en.pkl', 
                          spanish_lexicon='lexicon/es/es.pkl', 
                          french_lexicon='lexicon/fr/fr.pkl'):
    """compare english, spanish, and french lexicon geometries"""
    print("\n=== three language comparison ===")
    
    # load all lexicons
    with open(english_lexicon, 'rb') as f:
        english_data = pickle.load(f)
    
    with open(spanish_lexicon, 'rb') as f:
        spanish_data = pickle.load(f)
    
    with open(french_lexicon, 'rb') as f:
        french_data = pickle.load(f)
    
    # extract statistics
    english_masses = [data['mass'] for data in english_data.values()]
    spanish_masses = [data['mass'] for data in spanish_data.values()]
    french_masses = [data['mass'] for data in french_data.values()]
    
    english_phases = [data['phase'] for data in english_data.values()]
    spanish_phases = [data['phase'] for data in spanish_data.values()]
    french_phases = [data['phase'] for data in french_data.values()]
    
    # mass distribution comparison
    print(f"mass distribution comparison:")
    print(f"  english: {len(english_data):,} words, range {min(english_masses):.3f} - {max(english_masses):.3f}")
    print(f"  spanish: {len(spanish_data):,} words, range {min(spanish_masses):.3f} - {max(spanish_masses):.3f}")
    print(f"  french: {len(french_data):,} words, range {min(french_masses):.3f} - {max(french_masses):.3f}")
    
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
    
    print(f"\nmass distribution peaks:")
    print(f"  english: {english_peaks} peaks")
    print(f"  spanish: {spanish_peaks} peaks")
    print(f"  french: {french_peaks} peaks")
    
    # phase distribution comparison
    phase_bins = np.linspace(0, 2*np.pi, 17)
    english_phase_hist, _ = np.histogram(english_phases, bins=phase_bins)
    spanish_phase_hist, _ = np.histogram(spanish_phases, bins=phase_bins)
    french_phase_hist, _ = np.histogram(french_phases, bins=phase_bins)
    
    # significant phase clusters
    def count_clusters(hist):
        return sum(1 for count in hist if count > 1000)
    
    english_clusters = count_clusters(english_phase_hist)
    spanish_clusters = count_clusters(spanish_phase_hist)
    french_clusters = count_clusters(french_phase_hist)
    
    print(f"\nphase cluster comparison:")
    print(f"  english: {english_clusters} significant clusters")
    print(f"  spanish: {spanish_clusters} significant clusters")
    print(f"  french: {french_clusters} significant clusters")
    
    # similarity assessment
    en_es_mass_sim = 1 - abs(np.mean(english_masses) - np.mean(spanish_masses))
    en_fr_mass_sim = 1 - abs(np.mean(english_masses) - np.mean(french_masses))
    es_fr_mass_sim = 1 - abs(np.mean(spanish_masses) - np.mean(french_masses))
    
    en_es_phase_sim = 1 - abs(np.mean(english_phases) - np.mean(spanish_phases)) / (2*np.pi)
    en_fr_phase_sim = 1 - abs(np.mean(english_phases) - np.mean(french_phases)) / (2*np.pi)
    es_fr_phase_sim = 1 - abs(np.mean(spanish_phases) - np.mean(french_phases)) / (2*np.pi)
    
    print(f"\nsimilarity assessment:")
    print(f"  mass similarity:")
    print(f"    en-es: {en_es_mass_sim:.3f}")
    print(f"    en-fr: {en_fr_mass_sim:.3f}")
    print(f"    es-fr: {es_fr_mass_sim:.3f}")
    print(f"  phase similarity:")
    print(f"    en-es: {en_es_phase_sim:.3f}")
    print(f"    en-fr: {en_fr_phase_sim:.3f}")
    print(f"    es-fr: {es_fr_phase_sim:.3f}")
    
    # universal vs language-specific
    if english_clusters == spanish_clusters == french_clusters:
        print(f"\nresult: universal phase topology confirmed!")
        print(f"   all {len([english_clusters, spanish_clusters, french_clusters])} languages show {english_clusters} phase clusters")
    else:
        print(f"\nresult: language-specific phase signatures detected")
    
    if english_peaks == spanish_peaks == french_peaks:
        print(f"   universal mass distribution: all {english_peaks} peaks")
    else:
        print(f"   language-specific mass distributions: en({english_peaks}) es({spanish_peaks}) fr({french_peaks}) peaks")
    
    return {
        'english_peaks': english_peaks,
        'spanish_peaks': spanish_peaks,
        'french_peaks': french_peaks,
        'english_clusters': english_clusters,
        'spanish_clusters': spanish_clusters,
        'french_clusters': french_clusters,
        'mass_similarities': {'en_es': en_es_mass_sim, 'en_fr': en_fr_mass_sim, 'es_fr': es_fr_mass_sim},
        'phase_similarities': {'en_es': en_es_phase_sim, 'en_fr': en_fr_phase_sim, 'es_fr': es_fr_phase_sim}
    }

def main():
    """main french wordnet processing"""
    print("=== french wordnet orbihex processing ===")
    
    # create directories
    os.makedirs('dataset/fr', exist_ok=True)
    os.makedirs('lexicon/fr', exist_ok=True)
    
    # parse french wordnet
    parser = french_wordnet_parser()
    french_words = parser.parse_french_wordnet()
    parser.save_french_words()
    
    # build french lexicon
    builder = french_lexicon(french_words)
    french_lexicon_data = builder.build_french_lexicon()
    
    # analyze
    analysis = builder.analyze_lexicon()
    
    # save
    compression_ratio = builder.save_lexicon()
    
    # compare with english and spanish
    comparison = compare_three_languages()
    
    print(f"\n=== final french results ===")
    print(f"lexicon size: {len(french_lexicon_data)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return french_lexicon_data, analysis, comparison

if __name__ == "__main__":
    french_lexicon, analysis, comparison = main()
