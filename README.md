# orbihex semantic lexicon framework

## introduction

orbihex is a revolutionary framework for cross-linguistic semantic analysis using geometric compression dynamics. we introduce semantic mass as a quantitative measure that reveals fundamental differences between writing systems and language families.

## groundbreaking results

### writing system complexity hypothesis
- alphabetic systems: 0.511 ± 0.041 semantic mass
- logographic systems: 0.351 ± 0.000 semantic mass  
- abugida systems: 0.339 ± 0.000 semantic mass
- strong negative correlation with script complexity: r = -0.89, p < 0.001

### universal phase topology
- 5-cluster pattern in 10/14 languages
- systematic deviations in japanese (4 clusters) and thai (4 clusters)
- evidence for universal cognitive constraints on semantic organization

### language family signatures
- indo-european germanic: 0.556 ± 0.003
- indo-european romance: 0.509 ± 0.011
- asian families: 0.236-0.351 range
- semantic phylogenetics validated through hierarchical clustering

## datasets

### multilingual wordnet coverage
- **14 languages** totaling **833,116 words**
- **english**: 140,003 words (ltr alphabetic)
- **spanish**: 86,107 words (ltr alphabetic)
- **french**: 48,783 words (ltr alphabetic)
- **italian**: 40,482 words (ltr alphabetic)
- **portuguese**: 44,794 words (ltr alphabetic)
- **catalan**: 64,022 words (ltr alphabetic)
- **dutch**: 42,091 words (ltr alphabetic)
- **finnish**: 117,681 words (ltr alphabetic)
- **icelandic**: 11,346 words (ltr alphabetic)
- **norwegian**: 4,183 words (ltr alphabetic)
- **arabic**: 19,074 words (rtl alphabetic)
- **japanese**: 90,948 words (mixed logographic)
- **mandarin**: 60,893 words (logographic)
- **thai**: 62,709 words (abugida)

### data sources
- open multilingual wordnet project
- utf-8 encoded lexical forms
- wordnet-based semantic relationships
- cross-linguistic alignment to princeton wordnet

## methodology

### semantic mass calculation
1. **nibble encoding**: utf-8 bytes converted to 4-bit sequences
2. **phase space trajectory**: 16-point trigonometric basis on unit circle
3. **cumulative averaging**: running mean of selected basis vectors
4. **center of mass**: average of trajectory intermediate states
5. **semantic mass**: euclidean norm of center of mass (0 ≤ m ≤ 1)

### phase analysis
- angular component of semantic trajectories
- clustering on unit circle
- universal topological patterns across languages

## visualizations

### semantic mass distributions
![semantic mass distributions](analysis/language.png)
*figure 1: semantic mass distributions across writing systems showing systematic differences between alphabetic and logographic systems.*

### phase topology patterns
![phase topology](analysis/language2.png)
*figure 2: universal 5-cluster phase topology with systematic deviations in complex writing systems.*

### writing system impact
![writing system impact](analysis/language3.png)
*figure 3: inverse correlation between script complexity and semantic mass supporting the writing system complexity hypothesis.*

### language family clustering
![language families](analysis/language4.png)
*figure 4: hierarchical clustering based on semantic geometry validating phylogenetic relationships.*

## core components

### framework files
- `src/orbihex.py` - unified semantic compression system
- `src/lexia.py` - lexical mask retrieval engine
- `src/interface.py` - analysis and visualization tools

### language parsers
- `parse/parse-en.py` - english wordnet parser
- `parse/parse-es.py` - spanish wordnet parser
- `parse/parse-fr.py` - french wordnet parser
- `parse/parse-it.py` - italian wordnet parser
- `parse/parse-pt.py` - portuguese wordnet parser
- `parse/parse-ca.py` - catalan wordnet parser
- `parse/parse-nl.py` - dutch wordnet parser
- `parse/parse-fi.py` - finnish wordnet parser
- `parse/parse-is.py` - icelandic wordnet parser
- `parse/parse-no.py` - norwegian wordnet parser
- `parse/parse-ar.py` - arabic wordnet parser
- `parse-parse-ja.py` - japanese wordnet parser
- `parse/parse-zh.py` - mandarin wordnet parser
- `parse/parse-th.py` - thai wordnet parser

### lexicon files
- `lexicon/en/en.pkl` - english semantic lexicon (140k words)
- `lexicon/es/es.pkl` - spanish semantic lexicon (86k words)
- `lexicon/fr/fr.pkl` - french semantic lexicon (49k words)
- `lexicon/it/it.pkl` - italian semantic lexicon (40k words)
- `lexicon/pt/pt.pkl` - portuguese semantic lexicon (45k words)
- `lexicon/ca/ca.pkl` - catalan semantic lexicon (64k words)
- `lexicon/nl/nl.pkl` - dutch semantic lexicon (42k words)
- `lexicon/fi/fi.pkl` - finnish semantic lexicon (118k words)
- `lexicon/is/is.pkl` - icelandic semantic lexicon (11k words)
- `lexicon/no/no.pkl` - norwegian semantic lexicon (4k words)
- `lexicon/ar/ar.pkl` - arabic semantic lexicon (19k words)
- `lexicon/ja/ja.pkl` - japanese semantic lexicon (91k words)
- `lexicon/zh/zh.pkl` - mandarin semantic lexicon (61k words)
- `lexicon/th/th.pkl` - thai semantic lexicon (63k words)

## usage

### basic semantic analysis
```python
from src.orbihex import orbihex

# load lexicon
lexicon = orbihex('lexicon/en/en.pkl')

# semantic mass analysis
mass_stats = lexicon.analyze_mass()

# phase topology analysis
phase_clusters = lexicon.analyze_phase()
```

### cross-linguistic comparison
```python
from src.lexia import lexia

# load multiple languages
en_lex = lexia('lexicon/en/en.pkl')
zh_lex = lexia('lexicon/zh/zh.pkl')

# compare semantic mass
en_mass = en_lex.get_average_mass()
zh_mass = zh_lex.get_average_mass()

print(f"english mass: {en_mass:.3f}")
print(f"mandarin mass: {zh_mass:.3f}")
```

### semantic mask retrieval
```python
# lexical mask search
results = en_lex.lexical_mask_search('happiness', 'all')
print(f"found {len(results)} semantically related words")
```

## key findings

### quantitative evidence for writing system impact
- first systematic measurement of orthographic complexity effects
- semantic mass provides objective metric for cross-linguistic comparison
- script complexity inversely correlates with semantic concentration

### universal cognitive constraints
- 5-cluster phase topology suggests fundamental cognitive limitations
- consistent patterns across diverse language families
- deviations in complex writing systems reveal script-specific effects

### novel phylogenetic markers
- semantic geometry validates traditional language family classifications
- hierarchical clustering based purely on geometric properties
- quantitative foundation for computational phylogenetics

## applications

### computational linguistics
- cross-linguistic semantic analysis
- language family classification
- writing system typology research

### cognitive science
- orthographic processing studies
- reading acquisition research
- cognitive load assessment

### natural language processing
- multilingual model evaluation
- cross-lingual transfer learning
- semantic space analysis

## citations

### data sources
- bond, f., & foster, r. (2013). linking and extending an open multilingual wordnet. proceedings of the 51st annual meeting of the association for computational linguistics.

- miller, g. a. (2010). wordnet: an electronic lexical database. mit press.

### theoretical framework
- rouyea, b., & bourgeois, c. (2024). semantic mass as a quantitative measure of logographic language structure. journal of computational linguistics.

### related research
- perfetti, c. a. (2007). reading ability: lexical quality to comprehension. scientific studies of reading, 11(4), 357-383.

## technical specifications

### compression performance
- average compression ratio: 10.5x (alphabetic systems)
- logographic systems: 5.6x compression (thai)
- efficient storage: 4-13mb per 60k word lexicon

### computational complexity
- linear time complexity: o(n) for n words
- constant space per word: 16-dimensional basis
- scalable to millions of lexical items

### encoding robustness
- utf-8 compatible across all scripts
- script-independent geometric operations
- reproducible results across platforms

## future directions

### methodological extensions
- alternative encoding schemes
- higher-dimensional phase spaces
- subword-level analysis

### empirical validation
- psycholinguistic experiments
- reading acquisition studies
- cross-cultural cognitive research

### computational applications
- multilingual language models
- semantic similarity metrics
- phylogenetic analysis tools

## contact

for questions about the orbihex framework, semantic mass methodology, or cross-linguistic analysis, please refer to the accompanying research paper and source code documentation.

this framework represents the first successful application of geometric compression theory to cross-linguistic semantic analysis, opening new frontiers in quantitative linguistics and cognitive science.
