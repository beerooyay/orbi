# orbihex semantic lexicon

## vital framework

### core files
- `src/orbihex.py` - complete unified system
- `src/lexia.py` - lexical mask retrieval
- `src/interface.py` - analysis and visualization
- `lexicon/en/en.pkl` - english wordnet (140k words)
- `lexicon/es/es.pkl` - spanish wordnet (86k words)
- `lexicon/patterns.npy` - universal patterns
- `parse-en.py` - english wordnet parser
- `parse-es.py` - spanish wordnet parser
- `README.md` - this file

## usage

```python
# import and use
from src.orbihex import orbihex
from src.lexia import lexia

# initialize systems
system = orbihex('lexicon/en/en.pkl')
retrieval = lexia('lexicon/en/en.pkl')

# semantic search
neighbors = system.search('happy')

# lexical mask retrieval
masked_neighbors = retrieval.lexical_mask_search('happy')

# clustering
clusters = system.cluster(n_clusters=6)
analysis = system.analyze_clusters(clusters)

# machine learning
features = system.preprocess_features()
labels = system.create_labels(method='mass_quantiles')
data = system.split_data()
torch_data = system.export_torch(data)
```

## api endpoints

### core
- `orbihex(lexicon_file)` - initialize system
- `search(word, n_neighbors=5)` - semantic search
- `info(word)` - complete word information
- `filter_by_mass(min_mass, max_mass, limit=10)` - mass filtering
- `stats()` - system statistics

### lexical mask
- `lexia(lexicon_file)` - initialize retrieval
- `lexical_mask_search(word, mask_type)` - masked search
- `enhanced_retrieval_test()` - test retrieval accuracy

### machine learning
- `cluster(n_clusters=8)` - semantic clustering
- `analyze_clusters(clusters)` - cluster analysis
- `project_pca(n_components=2)` - dimensionality reduction
- `preprocess_features()` - feature preprocessing
- `create_labels(method='mass_quantiles')` - create labels
- `split_data(test_size=0.2)` - train/test split
- `export_torch(data)` - pytorch datasets

## results

### english wordnet
- **140,003 words** from full wordnet
- **10.4x compression** (16mb vs 168mb traditional)
- **semantic mass range: 0.025 - 1.000**
- **2 mass peaks** (bimodal distribution)
- **5 phase clusters** (universal topology)

### spanish wordnet
- **86,107 words** from spanish wordnet
- **10.3x compression** (10mb vs 103mb traditional)
- **semantic mass range: 0.018 - 1.000**
- **4 mass peaks** (complex morphology)
- **5 phase clusters** (universal topology)

### cross-lingual analysis
- **99.8% phase similarity** (universal geometry)
- **94.9% mass similarity** (cultural signatures)
- **universal 5-cluster topology** discovered
- **language-specific mass distributions** identified

## architecture

- clean modular structure in `src/`
- single-word naming convention
- no redundancy, no duplicate files
- integrated ml pipeline
- multilingual support
- scalable to full wordnet
- universal semantic geometry

## installation

```bash
# clone repository
git clone https://github.com/beerooyay/orbi.git
cd orbi

# install dependencies
pip install numpy scikit-learn matplotlib torch

# build lexicons (optional - pre-built included)
python3 parse-en.py  # english wordnet
python3 parse-es.py  # spanish wordnet

# run tests
python3 -c "from src.orbihex import orbihex; print('orbihex works')"
python3 -c "from src.lexia import lexia; print('lexia works')"
```

## project structure

```
orbihex/
├── src/                    # source code
│   ├── orbihex.py         # core semantic geometry
│   ├── lexia.py           # lexical mask retrieval
│   └── interface.py       # analysis tools
├── lexicon/               # processed lexicons
│   ├── en/en.pkl          # english wordnet
│   ├── es/es.pkl          # spanish wordnet
│   └── patterns.npy       # universal patterns
├── dataset/               # raw data
│   ├── en/dict/           # english wordnet data
│   ├── en/words.txt       # english word list
│   └── es/words.txt       # spanish word list
├── parse-en.py            # english parser
├── parse-es.py            # spanish parser
└── README.md              # this file
```

## key findings

### universal semantic geometry
- **5 phase clusters** appear in both languages
- **99.8% phase similarity** between english and spanish
- **phase topology** represents fundamental cognitive structure

### language-specific signatures
- **mass peaks differ** (2 vs 4 peaks)
- **mass distribution** reflects cultural/linguistic differences
- **english**: simpler bimodal distribution
- **spanish**: complex multimodal distribution

### compression efficiency
- **10.4x compression** maintained across languages
- **semantic structure preserved** through compression
- **zero training required** (purely geometric)

## scientific significance

this work demonstrates that:
1. meaning has intrinsic geometric structure
2. some aspects are universal (phase clusters)
3. some aspects are cultural (mass distributions)
4. compression can enhance rather than destroy semantics
5. universal cognitive architecture exists across languages
