#!/usr/bin/env python3
"""
norwegian wordnet parser for orbihex
extract norwegian words and build semantic lexicon
"""

import os
import pickle
import numpy as np
import binascii
from collections import defaultdict

class norwegian_wordnet_parser:
    def __init__(self, data_file='omw-data/wns/nor/wn-data-nob.tab'):
        self.data_file = data_file
        self.words = set()
        
    def parse_norwegian_wordnet(self):
        """parse norwegian wordnet data"""
        if not os.path.exists(self.data_file):
            print(f"norwegian wordnet not found: {self.data_file}")
            return []
        
        print(f"parsing norwegian wordnet from {self.data_file}...")
        
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
                
                # clean word: remove spaces, special chars, keep norwegian letters
                clean_word = ''.join(c for c in lemma.lower() if c.isalpha())
                
                if 2 <= len(clean_word) <= 20:
                    words.add(clean_word)
                
                if line_num % 500 == 0:
                    print(f"  processed {line_num:,} lines, found {len(words)} words")
        
        self.words = sorted(list(words))
        print(f"total norwegian words: {len(self.words)}")
        return self.words
    
    def save_norwegian_words(self, filename='dataset/no/words.txt'):
        """save norwegian word list"""
        with open(filename, 'w', encoding='utf-8') as f:
            for word in self.words:
                f.write(f"{word}\n")
        print(f"saved {len(self.words)} norwegian words to {filename}")

class norwegian_lexicon:
    def __init__(self, words, output_file='lexicon/no/no.pkl'):
        self.words = words
        self.output_file = output_file
        self.lexicon = {}
        
    def word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def build_norwegian_lexicon(self):
        """build norwegian lexicon"""
        print(f"building norwegian lexicon for {len(self.words)} words...")
        
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
            if processed % 500 == 0:
                print(f"processed {processed}/{len(self.words)} words ({processed/len(self.words)*100:.1f}%)")
        
        print(f"norwegian lexicon built: {len(self.lexicon)} words")
        return self.lexicon
    
    def save_lexicon(self):
        """save norwegian lexicon"""
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
        
        file_size = os.path.getsize(self.output_file)
        print(f"saved norwegian lexicon to {self.output_file} ({file_size:,} bytes)")
        
        # calculate compression
        traditional_size = len(self.lexicon) * 300 * 4
        compression_ratio = traditional_size / file_size
        
        print(f"traditional would be: {traditional_size:,} bytes")
        print(f"compression ratio: {compression_ratio:.1f}x")
        
        return compression_ratio
    
    def analyze_lexicon(self):
        """analyze norwegian lexicon"""
        if not self.lexicon:
            print("lexicon not built yet")
            return
        
        masses = [data['mass'] for data in self.lexicon.values()]
        phases = [data['phase'] for data in self.lexicon.values()]
        lengths = [data['length'] for data in self.lexicon.values()]
        
        print(f"\n=== norwegian lexicon analysis ===")
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

def compare_four_languages(english_lexicon='lexicon/en/en.pkl', 
                          spanish_lexicon='lexicon/es/es.pkl', 
                          french_lexicon='lexicon/fr/fr.pkl',
                          norwegian_lexicon='lexicon/no/no.pkl'):
    """compare english, spanish, french, and norwegian lexicon geometries"""
    print("\n=== four language comparison ===")
    
    # load all lexicons
    with open(english_lexicon, 'rb') as f:
        english_data = pickle.load(f)
    
    with open(spanish_lexicon, 'rb') as f:
        spanish_data = pickle.load(f)
    
    with open(french_lexicon, 'rb') as f:
        french_data = pickle.load(f)
    
    with open(norwegian_lexicon, 'rb') as f:
        norwegian_data = pickle.load(f)
    
    # extract statistics
    english_masses = [data['mass'] for data in english_data.values()]
    spanish_masses = [data['mass'] for data in spanish_data.values()]
    french_masses = [data['mass'] for data in french_data.values()]
    norwegian_masses = [data['mass'] for data in norwegian_data.values()]
    
    english_phases = [data['phase'] for data in english_data.values()]
    spanish_phases = [data['phase'] for data in spanish_data.values()]
    french_phases = [data['phase'] for data in french_data.values()]
    norwegian_phases = [data['phase'] for data in norwegian_data.values()]
    
    # mass distribution comparison
    print(f"mass distribution comparison:")
    print(f"  english: {len(english_data):,} words, range {min(english_masses):.3f} - {max(english_masses):.3f}")
    print(f"  spanish: {len(spanish_data):,} words, range {min(spanish_masses):.3f} - {max(spanish_masses):.3f}")
    print(f"  french: {len(french_data):,} words, range {min(french_masses):.3f} - {max(french_masses):.3f}")
    print(f"  norwegian: {len(norwegian_data):,} words, range {min(norwegian_masses):.3f} - {max(norwegian_masses):.3f}")
    
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
    
    print(f"\nmass distribution peaks:")
    print(f"  english: {english_peaks} peaks")
    print(f"  spanish: {spanish_peaks} peaks")
    print(f"  french: {french_peaks} peaks")
    print(f"  norwegian: {norwegian_peaks} peaks")
    
    # phase distribution comparison
    phase_bins = np.linspace(0, 2*np.pi, 17)
    english_phase_hist, _ = np.histogram(english_phases, bins=phase_bins)
    spanish_phase_hist, _ = np.histogram(spanish_phases, bins=phase_bins)
    french_phase_hist, _ = np.histogram(french_phases, bins=phase_bins)
    norwegian_phase_hist, _ = np.histogram(norwegian_phases, bins=phase_bins)
    
    # significant phase clusters
    def count_clusters(hist):
        return sum(1 for count in hist if count > 1000)
    
    english_clusters = count_clusters(english_phase_hist)
    spanish_clusters = count_clusters(spanish_phase_hist)
    french_clusters = count_clusters(french_phase_hist)
    norwegian_clusters = count_clusters(norwegian_phase_hist)
    
    print(f"\nphase cluster comparison:")
    print(f"  english: {english_clusters} significant clusters")
    print(f"  spanish: {spanish_clusters} significant clusters")
    print(f"  french: {french_clusters} significant clusters")
    print(f"  norwegian: {norwegian_clusters} significant clusters")
    
    # similarity assessment
    en_es_mass_sim = 1 - abs(np.mean(english_masses) - np.mean(spanish_masses))
    en_fr_mass_sim = 1 - abs(np.mean(english_masses) - np.mean(french_masses))
    en_no_mass_sim = 1 - abs(np.mean(english_masses) - np.mean(norwegian_masses))
    es_fr_mass_sim = 1 - abs(np.mean(spanish_masses) - np.mean(french_masses))
    es_no_mass_sim = 1 - abs(np.mean(spanish_masses) - np.mean(norwegian_masses))
    fr_no_mass_sim = 1 - abs(np.mean(french_masses) - np.mean(norwegian_masses))
    
    en_es_phase_sim = 1 - abs(np.mean(english_phases) - np.mean(spanish_phases)) / (2*np.pi)
    en_fr_phase_sim = 1 - abs(np.mean(english_phases) - np.mean(french_phases)) / (2*np.pi)
    en_no_phase_sim = 1 - abs(np.mean(english_phases) - np.mean(norwegian_phases)) / (2*np.pi)
    es_fr_phase_sim = 1 - abs(np.mean(spanish_phases) - np.mean(french_phases)) / (2*np.pi)
    es_no_phase_sim = 1 - abs(np.mean(spanish_phases) - np.mean(norwegian_phases)) / (2*np.pi)
    fr_no_phase_sim = 1 - abs(np.mean(french_phases) - np.mean(norwegian_phases)) / (2*np.pi)
    
    print(f"\nsimilarity assessment:")
    print(f"  mass similarity:")
    print(f"    en-es: {en_es_mass_sim:.3f}")
    print(f"    en-fr: {en_fr_mass_sim:.3f}")
    print(f"    en-no: {en_no_mass_sim:.3f}")
    print(f"    es-fr: {es_fr_mass_sim:.3f}")
    print(f"    es-no: {es_no_mass_sim:.3f}")
    print(f"    fr-no: {fr_no_mass_sim:.3f}")
    print(f"  phase similarity:")
    print(f"    en-es: {en_es_phase_sim:.3f}")
    print(f"    en-fr: {en_fr_phase_sim:.3f}")
    print(f"    en-no: {en_no_phase_sim:.3f}")
    print(f"    es-fr: {es_fr_phase_sim:.3f}")
    print(f"    es-no: {es_no_phase_sim:.3f}")
    print(f"    fr-no: {fr_no_phase_sim:.3f}")
    
    # universal vs language-specific
    if english_clusters == spanish_clusters == french_clusters == norwegian_clusters:
        print(f"\nresult: universal phase topology confirmed!")
        print(f"   all {len([english_clusters, spanish_clusters, french_clusters, norwegian_clusters])} languages show {english_clusters} phase clusters")
    else:
        print(f"\nresult: language-specific phase signatures detected")
    
    peak_counts = [english_peaks, spanish_peaks, french_peaks, norwegian_peaks]
    if len(set(peak_counts)) == 1:
        print(f"   universal mass distribution: all {english_peaks} peaks")
    else:
        print(f"   language-specific mass distributions: en({english_peaks}) es({spanish_peaks}) fr({french_peaks}) no({norwegian_peaks}) peaks")
    
    return {
        'english_peaks': english_peaks,
        'spanish_peaks': spanish_peaks,
        'french_peaks': french_peaks,
        'norwegian_peaks': norwegian_peaks,
        'english_clusters': english_clusters,
        'spanish_clusters': spanish_clusters,
        'french_clusters': french_clusters,
        'norwegian_clusters': norwegian_clusters,
        'mass_similarities': {
            'en_es': en_es_mass_sim, 'en_fr': en_fr_mass_sim, 'en_no': en_no_mass_sim,
            'es_fr': es_fr_mass_sim, 'es_no': es_no_mass_sim, 'fr_no': fr_no_mass_sim
        },
        'phase_similarities': {
            'en_es': en_es_phase_sim, 'en_fr': en_fr_phase_sim, 'en_no': en_no_phase_sim,
            'es_fr': es_fr_phase_sim, 'es_no': es_no_phase_sim, 'fr_no': fr_no_phase_sim
        }
    }

def main():
    """main norwegian wordnet processing"""
    print("=== norwegian wordnet orbihex processing ===")
    
    # create directories
    os.makedirs('dataset/no', exist_ok=True)
    os.makedirs('lexicon/no', exist_ok=True)
    
    # parse norwegian wordnet
    parser = norwegian_wordnet_parser()
    norwegian_words = parser.parse_norwegian_wordnet()
    parser.save_norwegian_words()
    
    # build norwegian lexicon
    builder = norwegian_lexicon(norwegian_words)
    norwegian_lexicon_data = builder.build_norwegian_lexicon()
    
    # analyze
    analysis = builder.analyze_lexicon()
    
    # save
    compression_ratio = builder.save_lexicon()
    
    # compare with english, spanish, and french
    comparison = compare_four_languages()
    
    print(f"\n=== final norwegian results ===")
    print(f"lexicon size: {len(norwegian_lexicon_data)} words")
    print(f"compression ratio: {compression_ratio:.1f}x")
    print(f"mass range: {analysis['mass_range'][0]:.3f} - {analysis['mass_range'][1]:.3f}")
    
    return norwegian_lexicon_data, analysis, comparison

if __name__ == "__main__":
    norwegian_lexicon, analysis, comparison = main()
