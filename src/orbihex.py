#!/usr/bin/env python3
"""
orbihex semantic lexicon
single unified system for semantic geometry and compression
"""

import os
import pickle
import numpy as np
import binascii
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class orbihex:
    """orbihex semantic lexicon system"""
    
    def __init__(self, lexicon_file='lexicon/en/en.pkl'):
        """initialize orbihex system"""
        self.lexicon_file = lexicon_file
        self.lexicon = None
        self.words = []
        self.com_matrix = None
        self.mass_vector = None
        self.phase_vector = None
        self.nn_model = None
        self.kmeans = None
        self.pca = None
        self.scalers = {}
        
        self._load_or_build()
    
    def _load_or_build(self):
        """load existing lexicon or build new one"""
        if os.path.exists(self.lexicon_file):
            print(f"loading lexicon from {self.lexicon_file}")
            self._load_lexicon()
        else:
            print("building new lexicon...")
            self._build_lexicon()
            self._save_lexicon()
        
        self._extract_arrays()
        self._build_neighbor_model()
        print(f"orbihex ready: {len(self.words)} words loaded")
    
    def _load_lexicon(self):
        """load lexicon from file"""
        with open(self.lexicon_file, 'rb') as f:
            self.lexicon = pickle.load(f)
    
    def _save_lexicon(self):
        """save lexicon to file"""
        os.makedirs(os.path.dirname(self.lexicon_file), exist_ok=True)
        with open(self.lexicon_file, 'wb') as f:
            pickle.dump(self.lexicon, f)
    
    def _extract_arrays(self):
        """extract numpy arrays from lexicon"""
        self.words = list(self.lexicon.keys())
        self.com_matrix = np.array([self.lexicon[w]['com'] for w in self.words])
        self.mass_vector = np.array([self.lexicon[w]['mass'] for w in self.words])
        self.phase_vector = np.array([self.lexicon[w]['phase'] for w in self.words])
    
    def _build_neighbor_model(self):
        """build nearest neighbors model"""
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nn_model.fit(self.com_matrix)
    
    def _build_lexicon(self):
        """build complete lexicon"""
        words = self._get_word_list()
        
        # phase basis for nibble mapping
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        basis = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        self.lexicon = {}
        for word in words:
            nibbles = self._word_to_nibbles(word)
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
    
    def _get_word_list(self):
        """get comprehensive word list"""
        emotion_words = [
            "happy", "sad", "angry", "anxious", "joyful", "miserable", "furious", "worried",
            "excited", "depressed", "enraged", "nervous", "cheerful", "lonely", "hostile",
            "afraid", "content", "gloomy", "bitter", "tense", "peaceful", "tired", "annoyed",
            "scared", "proud", "hopeless", "resentful", "uneasy", "grateful", "broken",
            "disgusted", "calm", "lost", "hateful", "restless", "bright", "grief", "vengeful",
            "distracted", "smiling", "regret", "fearful", "blessed", "sorrow", "aggravated",
            "timid", "energetic", "helpless", "seething", "relaxed", "numb", "fedup", "racing",
            "ecstatic", "devastated", "appalled", "thrilled", "horrified", "elated", "crushed",
            "outraged", "delighted", "despair", "euphoric", "melancholy", "irate", "jubilant",
            "forlorn", "blissful", "anguished", "contented", "serene", "wistful"
        ]
        
        concept_words = [
            "love", "hate", "death", "life", "time", "space", "mind", "body", "soul", "heart",
            "brain", "thought", "dream", "reality", "truth", "lie", "hope", "fear", "pain",
            "pleasure", "joy", "suffering", "peace", "war", "home", "family", "friend", "enemy",
            "stranger", "child", "adult", "man", "woman", "person", "human", "animal", "plant",
            "tree", "flower", "water", "fire", "earth", "air", "sky", "sun", "moon", "star",
            "night", "day", "morning", "evening", "light", "dark", "shadow", "mirror", "window",
            "door", "house", "city", "town", "village", "country", "world", "universe",
            "nature", "culture", "society", "civilization", "history", "future", "past",
            "present", "moment", "eternity", "infinity", "science", "art", "music", "poetry",
            "literature", "philosophy", "religion", "spirituality", "consciousness", "subconscious",
            "memory", "imagination", "creativity", "intelligence", "wisdom", "knowledge",
            "understanding", "awareness", "perception", "sensation", "freedom", "justice",
            "equality", "liberty", "democracy", "tyranny", "power", "authority", "control"
        ]
        
        action_words = [
            "run", "walk", "jump", "fly", "swim", "climb", "fall", "rise", "sit", "stand",
            "sleep", "wake", "eat", "drink", "breathe", "think", "know", "learn", "teach",
            "speak", "listen", "read", "write", "sing", "dance", "play", "work", "rest",
            "fight", "help", "give", "take", "make", "break", "build", "destroy", "create",
            "imagine", "remember", "forget", "believe", "doubt", "trust", "betray", "love",
            "hate", "feel", "touch", "see", "hear", "smell", "taste", "perceive", "understand",
            "confuse", "clarify", "obscure", "reveal", "hide", "connect", "disconnect", "unite",
            "divide", "merge", "separate", "grow", "shrink", "expand", "contract", "stretch",
            "compress", "accelerate", "decelerate", "stop", "start", "continue", "pause"
        ]
        
        descriptive_words = [
            "good", "bad", "beautiful", "ugly", "big", "small", "hot", "cold", "fast", "slow",
            "hard", "soft", "bright", "dark", "heavy", "light", "strong", "weak", "young",
            "old", "new", "ancient", "modern", "primitive", "simple", "complex", "easy",
            "difficult", "clear", "unclear", "clean", "dirty", "brave", "cowardly", "confident",
            "insecure", "proud", "ashamed", "humble", "arrogant", "kind", "cruel", "gentle",
            "harsh", "patient", "impatient", "tolerant", "intolerant", "honest", "dishonest",
            "truthful", "deceitful", "loyal", "disloyal", "faithful", "unfaithful", "smart",
            "stupid", "intelligent", "dumb", "clever", "foolish", "wise", "ignorant", "educated",
            "uneducated", "skilled", "unskilled", "talented", "untalented", "gifted", "ordinary"
        ]
        
        # combine and filter
        all_words = list(set(emotion_words + concept_words + action_words + descriptive_words))
        filtered_words = [w for w in all_words if 2 <= len(w) <= 20 and w.isalpha()]
        
        return filtered_words
    
    def _word_to_nibbles(self, word):
        """convert word to nibble sequence"""
        bytes_word = word.encode("utf-8")
        hex_word = binascii.hexlify(bytes_word).decode("ascii")
        return [int(hex_word[i:i+1], 16) for i in range(len(hex_word))]
    
    def search(self, word, n_neighbors=5):
        """semantic search - find neighbors of word"""
        if word not in self.lexicon:
            return []
        
        word_idx = self.words.index(word)
        word_com = self.com_matrix[word_idx].reshape(1, -1)
        
        distances, indices = self.nn_model.kneighbors(word_com)
        neighbors = [self.words[i] for i in indices[0][1:]]
        
        return list(zip(neighbors, distances[0][1:]))
    
    def info(self, word):
        """get complete word information"""
        if word not in self.lexicon:
            return None
        
        data = self.lexicon[word]
        return {
            'word': word,
            'com': data['com'],
            'mass': data['mass'],
            'phase': data['phase'],
            'length': data['length'],
            'neighbors': self.search(word, 5)
        }
    
    def filter_by_mass(self, min_mass=None, max_mass=None, limit=10):
        """filter words by semantic mass range"""
        if min_mass is None:
            min_mass = np.min(self.mass_vector)
        if max_mass is None:
            max_mass = np.max(self.mass_vector)
        
        mask = (self.mass_vector >= min_mass) & (self.mass_vector <= max_mass)
        candidates = [(self.words[i], self.mass_vector[i]) for i in range(len(self.words)) if mask[i]]
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]
    
    def cluster(self, n_clusters=8):
        """build semantic clusters"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(self.com_matrix)
        
        clusters = {}
        for i, word in enumerate(self.words):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(word)
        
        return clusters
    
    def analyze_clusters(self, clusters):
        """analyze cluster characteristics"""
        analysis = {}
        for cluster_id, words in clusters.items():
            masses = [self.lexicon[w]['mass'] for w in words]
            analysis[cluster_id] = {
                'size': len(words),
                'avg_mass': np.mean(masses),
                'mass_std': np.std(masses),
                'sample_words': words[:5]
            }
        return analysis
    
    def project_pca(self, n_components=2):
        """project to lower dimensions for visualization"""
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(self.com_matrix)
    
    def preprocess_features(self, scale_com=True, scale_mass=True, scale_phase=True):
        """preprocess features for machine learning"""
        features = []
        feature_names = []
        
        # com coordinates
        if scale_com:
            self.scalers['com'] = StandardScaler()
            com_scaled = self.scalers['com'].fit_transform(self.com_matrix)
            features.append(com_scaled)
            feature_names.extend(['com_x', 'com_y'])
        else:
            features.append(self.com_matrix)
            feature_names.extend(['com_x', 'com_y'])
        
        # semantic mass
        if scale_mass:
            self.scalers['mass'] = MinMaxScaler()
            mass_scaled = self.scalers['mass'].fit_transform(self.mass_vector.reshape(-1, 1))
            features.append(mass_scaled)
            feature_names.append('mass')
        else:
            features.append(self.mass_vector.reshape(-1, 1))
            feature_names.append('mass')
        
        # phase
        if scale_phase:
            self.scalers['phase'] = MinMaxScaler()
            phase_scaled = self.scalers['phase'].fit_transform(self.phase_vector.reshape(-1, 1))
            features.append(phase_scaled)
            feature_names.append('phase')
        else:
            features.append(self.phase_vector.reshape(-1, 1))
            feature_names.append('phase')
        
        self.feature_matrix = np.hstack(features)
        self.feature_names = feature_names
        
        return self.feature_matrix
    
    def create_labels(self, method='mass_quantiles', n_classes=3):
        """create supervised learning labels"""
        if method == 'mass_quantiles':
            quantiles = np.percentile(self.mass_vector, 
                                     np.linspace(0, 100, n_classes + 1)[1:-1])
            labels = np.digitize(self.mass_vector, quantiles)
            label_names = [f'low_mass', 'medium_mass', 'high_mass'][:n_classes]
        
        elif method == 'phase_bins':
            phase_bins = np.linspace(0, 2*np.pi, n_classes + 1)
            labels = np.digitize(self.phase_vector, phase_bins[:-1])
            label_names = [f'phase_{i}' for i in range(n_classes)]
        
        elif method == 'semantic_clusters':
            if self.kmeans is None:
                self.cluster(n_classes)
            labels = self.kmeans.labels_
            label_names = [f'semantic_cluster_{i}' for i in range(n_classes)]
        
        else:
            raise ValueError(f"unknown method: {method}")
        
        self.labels = labels
        self.label_names = label_names
        self.label_method = method
        
        return labels
    
    def split_data(self, test_size=0.2, random_state=42):
        """train/test split for machine learning"""
        if not hasattr(self, 'feature_matrix'):
            self.preprocess_features()
        
        if not hasattr(self, 'labels'):
            self.create_labels()
        
        X_train, X_test, y_train, y_test, words_train, words_test = train_test_split(
            self.feature_matrix, self.labels, self.words, 
            test_size=test_size, random_state=random_state, stratify=self.labels
        )
        
        return {
            'train': {'X': X_train, 'y': y_train, 'words': words_train},
            'test': {'X': X_test, 'y': y_test, 'words': words_test}
        }
    
    def export_torch(self, data, batch_size=32):
        """export pytorch datasets"""
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            
            train_dataset = TensorDataset(
                torch.tensor(data['train']['X'], dtype=torch.float32),
                torch.tensor(data['train']['y'], dtype=torch.long)
            )
            
            test_dataset = TensorDataset(
                torch.tensor(data['test']['X'], dtype=torch.float32),
                torch.tensor(data['test']['y'], dtype=torch.long)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return {
                'train_loader': train_loader,
                'test_loader': test_loader,
                'feature_names': self.feature_names,
                'label_names': self.label_names,
                'n_classes': len(self.label_names)
            }
            
        except ImportError:
            print("pytorch not available")
            return None
    
    def stats(self):
        """get lexicon statistics"""
        return {
            'total_words': len(self.words),
            'mass_range': (float(np.min(self.mass_vector)), float(np.max(self.mass_vector))),
            'avg_mass': float(np.mean(self.mass_vector)),
            'phase_coverage': (float(np.min(self.phase_vector)), float(np.max(self.phase_vector))),
            'compression_ratio': len(self.words) * 300 * 4 / os.path.getsize(self.lexicon_file)
        }
    
    def demo(self):
        """demonstration of orbihex capabilities"""
        print("=== orbihex semantic search demo ===")
        
        # semantic search examples
        test_words = ['happy', 'death', 'love', 'time', 'consciousness']
        for word in test_words:
            if word in self.lexicon:
                info = self.info(word)
                neighbors = info['neighbors']
                print(f"\n{word} (mass={info['mass']:.3f}):")
                for neighbor, dist in neighbors[:3]:
                    print(f"  {neighbor} ({dist:.3f})")
        
        # mass filtering examples
        print(f"\nhigh mass words:")
        high_mass = self.filter_by_mass(min_mass=np.percentile(self.mass_vector, 90), limit=5)
        for word, mass in high_mass:
            print(f"  {word} ({mass:.3f})")
        
        print(f"\nlow mass words:")
        low_mass = self.filter_by_mass(max_mass=np.percentile(self.mass_vector, 10), limit=5)
        for word, mass in low_mass:
            print(f"  {word} ({mass:.3f})")
        
        # clustering analysis
        print(f"\n=== clustering analysis ===")
        clusters = self.cluster(n_clusters=6)
        cluster_analysis = self.analyze_clusters(clusters)
        
        for cluster_id, analysis in cluster_analysis.items():
            print(f"cluster {cluster_id}: {analysis['size']} words, avg mass = {analysis['avg_mass']:.3f}")
            print(f"  sample: {analysis['sample_words']}")
        
        # statistics
        stats_data = self.stats()
        print(f"\n=== orbihex statistics ===")
        print(f"total words: {stats_data['total_words']}")
        print(f"mass range: {stats_data['mass_range'][0]:.3f} - {stats_data['mass_range'][1]:.3f}")
        print(f"compression ratio: {stats_data['compression_ratio']:.1f}x")

def main():
    """main entry point"""
    orbihex_system = orbihex()
    orbihex_system.demo()
    return orbihex_system

if __name__ == "__main__":
    main()
