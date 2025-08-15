import random
import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# @dataclass
# class TestResult:
#     """Container for test results"""
#     query: str
#     perturbation_type: str
#     original_candidates: List[str]
#     perturbed_candidates: List[str]
#     original_scores: List[float]
#     perturbed_scores: List[float]
#     similarity_score: float
#     rank_correlation: float
#     metadata: Dict

# class HybridIRTester:
#     """Comprehensive testing framework for hybrid IR models"""
    
#     def __init__(self, ir_pipeline: Callable, ground_truth: Optional[Dict] = None):
#         """
#         Initialize the tester
        
#         Args:
#             ir_pipeline: Function that takes query and returns (candidates, scores)
#             ground_truth: Dict mapping queries to relevant document IDs for evaluation
#         """
#         self.ir_pipeline = ir_pipeline
#         self.ground_truth = ground_truth or {}
#         self.test_results = []
        
#     # --- Query Perturbation Methods ---
    
#     def synonym_perturbation(self, query: str, intensity: float = 0.3) -> str:
#         """Replace words with synonyms using WordNet"""
#         try:
#             import nlpaug.augmenter.word as naw
#             aug = naw.SynonymAug(aug_src='wordnet', aug_p=intensity)
#             return aug.augment(query)
#         except ImportError:
#             # Fallback: simple synonym replacement
#             synonyms = {
#                 'mortgage': 'home loan', 'bank': 'financial institution',
#                 'loan': 'credit', 'interest': 'rate', 'apply': 'request'
#             }
#             words = query.split()
#             for i, word in enumerate(words):
#                 if word.lower() in synonyms and random.random() < intensity:
#                     words[i] = synonyms[word.lower()]
#             return ' '.join(words)
    
#     def typo_perturbation(self, query: str, intensity: float = 0.1) -> str:
#         """Introduce keyboard typos"""
#         keyboard_map = {
#             'a': 'sq', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
#             'f': 'drtgvc', 'g': 'ftyb', 'h': 'gtyujn', 'i': 'ujko', 'j': 'huikm',
#             'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
#             'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'refgy',
#             'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
#             'z': 'asx'
#         }
        
#         chars = list(query.lower())
#         for i, char in enumerate(chars):
#             if char in keyboard_map and random.random() < intensity:
#                 chars[i] = random.choice(keyboard_map[char])
#         return ''.join(chars)
    
#     def paraphrase_perturbation(self, query: str) -> str:
#         """Paraphrase using rule-based transformations"""
#         # Simple rule-based paraphrasing
#         transformations = [
#             (r"How (can|do) I (.+)\?", r"I need to \2"),
#             (r"What (is|are) (.+)\?", r"Tell me about \2"),
#             (r"Where (is|are) (.+)\?", r"Location of \2"),
#             (r"When (.+)\?", r"Time for \1"),
#         ]
        
#         import re
#         result = query
#         for pattern, replacement in transformations:
#             result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
#         return result if result != query else f"Please help me with: {query}"
    
#     def length_perturbation(self, query: str, mode: str = 'truncate') -> str:
#         """Modify query length"""
#         words = query.split()
#         if mode == 'truncate' and len(words) > 3:
#             return ' '.join(words[:max(2, len(words)//2)])
#         elif mode == 'expand':
#             expansions = ['please', 'help me', 'I need information about', 'can you tell me']
#             return f"{random.choice(expansions)} {query}"
#         return query
    
#     def word_order_shuffle(self, query: str, intensity: float = 0.5) -> str:
#         """Shuffle word order partially"""
#         words = query.split()
#         if len(words) <= 2:
#             return query
        
#         # Shuffle only a portion of words
#         shuffle_count = max(1, int(len(words) * intensity))
#         indices = random.sample(range(len(words)), shuffle_count)
#         shuffled_words = [words[i] for i in indices]
#         random.shuffle(shuffled_words)
        
#         result = words.copy()
#         for i, idx in enumerate(indices):
#             result[idx] = shuffled_words[i]
#         return ' '.join(result)
    
#     def noise_injection(self, query: str, noise_type: str = 'irrelevant') -> str:
#         """Inject various types of noise"""
#         if noise_type == 'irrelevant':
#             noise_phrases = [
#                 "and the weather is nice",
#                 "by the way hello",
#                 "random information here",
#                 "also I like coffee"
#             ]
#             return f"{query} {random.choice(noise_phrases)}"
#         elif noise_type == 'repetition':
#             words = query.split()
#             if words:
#                 repeat_word = random.choice(words)
#                 return f"{query} {repeat_word} {repeat_word}"
#         elif noise_type == 'punctuation':
#             return query.replace(' ', '...') + "???"
#         return query
    
#     def missing_terms(self, query: str, intensity: float = 0.3) -> str:
#         """Remove important terms"""
#         words = query.split()
#         if len(words) <= 2:
#             return query
        
#         remove_count = max(1, int(len(words) * intensity))
#         keep_indices = random.sample(range(len(words)), len(words) - remove_count)
#         return ' '.join([words[i] for i in sorted(keep_indices)])
    
#     def encoding_corruption(self, query: str) -> str:
#         """Simulate encoding issues"""
#         corruptions = [
#             lambda x: x.encode('utf-8', errors='replace').decode('utf-8', errors='replace'),
#             lambda x: x.replace('é', 'e').replace('ñ', 'n').replace('ü', 'u'),
#             lambda x: ''.join(c if ord(c) < 128 else '?' for c in x)
#         ]
#         return random.choice(corruptions)(query)
    
    
#     # --- End-to-End Robustness Testing ---
    
#     def run_perturbation_tests(self, queries: List[str], 
#                              perturbation_intensities: Dict[str, List[float]] = None) -> List[TestResult]:
#         """Run comprehensive perturbation tests"""
#         if perturbation_intensities is None:
#             perturbation_intensities = {
#                 'synonym': [0.1, 0.3, 0.5],
#                 'typo': [0.05, 0.1, 0.2],
#                 'length_truncate': [1.0],
#                 'length_expand': [1.0],
#                 'word_shuffle': [0.3, 0.5, 0.7],
#                 'noise': [1.0],
#                 'missing': [0.2, 0.4],
#                 'encoding': [1.0]
#             }
        
#         test_results = []
        
#         for query in queries:
#             logger.info(f"Testing query: {query}")
#             original_candidates, original_scores = self.ir_pipeline(query)
            
#             # Test each perturbation type
#             for pert_type, intensities in perturbation_intensities.items():
#                 for intensity in intensities:
#                     perturbed_query = self._apply_perturbation(query, pert_type, intensity)
#                     perturbed_candidates, perturbed_scores = self.ir_pipeline(perturbed_query)
                    
#                     # Calculate similarity metrics
#                     similarity_score = self._calculate_result_similarity(
#                         original_candidates, perturbed_candidates,
#                         original_scores, perturbed_scores
#                     )
                    
#                     rank_correlation = self._calculate_rank_correlation(
#                         original_candidates, perturbed_candidates,
#                         original_scores, perturbed_scores
#                     )
                    
#                     result = TestResult(
#                         query=query,
#                         perturbation_type=f"{pert_type}_{intensity}",
#                         original_candidates=original_candidates,
#                         perturbed_candidates=perturbed_candidates,
#                         original_scores=original_scores,
#                         perturbed_scores=perturbed_scores,
#                         similarity_score=similarity_score,
#                         rank_correlation=rank_correlation,
#                         metadata={
#                             'perturbed_query': perturbed_query,
#                             'intensity': intensity
#                         }
#                     )
                    
#                     test_results.append(result)
        
#         self.test_results.extend(test_results)
#         return test_results
    
#     def test_ood_domain_performance(self, ood_queries: List[str], 
#                                   expected_performance_drop: float = 0.2) -> Dict:
#         """Test out-of-domain query performance"""
#         results = {'queries': [], 'performance_metrics': []}
        
#         for query in ood_queries:
#             candidates, scores = self.ir_pipeline(query)
            
#             # Calculate various metrics
#             avg_score = np.mean(scores) if scores else 0.0
#             score_variance = np.var(scores) if scores else 0.0
#             num_candidates = len(candidates)
            
#             results['queries'].append(query)
#             results['performance_metrics'].append({
#                 'avg_score': avg_score,
#                 'score_variance': score_variance,
#                 'num_candidates': num_candidates,
#                 'query': query
#             })
        
#         # Compare with baseline performance if available
#         if hasattr(self, 'baseline_performance'):
#             for i, metrics in enumerate(results['performance_metrics']):
#                 performance_drop = (self.baseline_performance['avg_score'] - metrics['avg_score']) / self.baseline_performance['avg_score']
#                 metrics['performance_drop'] = performance_drop
#                 metrics['meets_expectation'] = performance_drop <= expected_performance_drop
        
#         return results
    
#     def test_multilingual_robustness(self, multilingual_queries: Dict[str, List[str]]) -> Dict:
#         """Test performance across different languages"""
#         results = {}
        
#         for language, queries in multilingual_queries.items():
#             lang_results = []
#             for query in queries:
#                 candidates, scores = self.ir_pipeline(query)
#                 lang_results.append({
#                     'query': query,
#                     'num_candidates': len(candidates),
#                     'avg_score': np.mean(scores) if scores else 0.0,
#                     'max_score': max(scores) if scores else 0.0
#                 })
#             results[language] = lang_results
        
#         return results
    
#     def test_adversarial_queries(self, base_queries: List[str]) -> Dict:
#         """Test against adversarially crafted queries"""
#         adversarial_patterns = [
#             lambda q: q + " " + " ".join(q.split()[::-1]),  # Query + reverse
#             lambda q: " ".join([word[::-1] for word in q.split()]),  # Reverse each word
#             lambda q: q.replace(" ", ""),  # Remove spaces
#             lambda q: q.upper(),  # All caps
#             lambda q: q.lower(),  # All lowercase
#             lambda q: "".join([c for c in q if c.isalnum() or c.isspace()]),  # Remove special chars
#         ]
        
#         results = {}
#         for i, pattern in enumerate(adversarial_patterns):
#             pattern_results = []
#             for query in base_queries:
#                 adv_query = pattern(query)
#                 candidates, scores = self.ir_pipeline(adv_query)
                
#                 # Compare with original
#                 orig_candidates, orig_scores = self.ir_pipeline(query)
#                 similarity = self._calculate_result_similarity(
#                     orig_candidates, candidates, orig_scores, scores
#                 )
                
#                 pattern_results.append({
#                     'original_query': query,
#                     'adversarial_query': adv_query,
#                     'similarity': similarity,
#                     'num_candidates': len(candidates)
#                 })
            
#             results[f'pattern_{i}'] = pattern_results
        
#         return results
    
#     # --- Performance Analysis and Reporting ---
    
#     def analyze_failure_modes(self, threshold: float = 0.5) -> Dict:
#         """Analyze common failure patterns in test results"""
#         if not self.test_results:
#             return {"error": "No test results available"}
        
#         failures = [r for r in self.test_results if r.similarity_score < threshold]
        
#         failure_analysis = {
#             'total_tests': len(self.test_results),
#             'failure_count': len(failures),
#             'failure_rate': len(failures) / len(self.test_results),
#             'failure_by_perturbation': defaultdict(int),
#             'most_vulnerable_queries': defaultdict(int),
#             'common_patterns': []
#         }
        
#         for failure in failures:
#             pert_type = failure.perturbation_type.split('_')[0]
#             failure_analysis['failure_by_perturbation'][pert_type] += 1
#             failure_analysis['most_vulnerable_queries'][failure.query] += 1
        
#         return dict(failure_analysis)
    
#     def generate_robustness_report(self, output_file: str = None) -> Dict:
#         """Generate comprehensive robustness report"""
#         if not self.test_results:
#             return {"error": "No test results available"}
        
#         report = {
#             'summary': {
#                 'total_tests': len(self.test_results),
#                 'avg_similarity': np.mean([r.similarity_score for r in self.test_results]),
#                 'avg_rank_correlation': np.mean([r.rank_correlation for r in self.test_results]),
#                 'min_similarity': min([r.similarity_score for r in self.test_results]),
#                 'max_similarity': max([r.similarity_score for r in self.test_results])
#             },
#             'perturbation_analysis': {},
#             'query_analysis': {},
#             'recommendations': []
#         }
        
#         # Analyze by perturbation type
#         by_perturbation = defaultdict(list)
#         for result in self.test_results:
#             pert_type = result.perturbation_type.split('_')[0]
#             by_perturbation[pert_type].append(result.similarity_score)
        
#         for pert_type, scores in by_perturbation.items():
#             report['perturbation_analysis'][pert_type] = {
#                 'avg_similarity': np.mean(scores),
#                 'std_similarity': np.std(scores),
#                 'min_similarity': min(scores),
#                 'robustness_score': np.mean([1 if s > 0.7 else 0 for s in scores])
#             }
        
#         # Generate recommendations
#         weak_perturbations = [k for k, v in report['perturbation_analysis'].items() 
#                             if v['avg_similarity'] < 0.6]
        
#         if 'typo' in weak_perturbations:
#             report['recommendations'].append("Consider adding spell correction preprocessing")
#         if 'synonym' in weak_perturbations:
#             report['recommendations'].append("Improve synonym handling in embeddings")
#         if 'word_shuffle' in weak_perturbations:
#             report['recommendations'].append("Enhance position-independent matching")
        
#         if output_file:
#             with open(output_file, 'w') as f:
#                 json.dump(report, f, indent=2)
        
#         return report
    
    
    
#     # --- Helper Methods ---
    
#     def _apply_perturbation(self, query: str, pert_type: str, intensity: float) -> str:
#         """Apply specific perturbation to query"""
#         perturbation_map = {
#             'synonym': lambda q, i: self.synonym_perturbation(q, i),
#             'typo': lambda q, i: self.typo_perturbation(q, i),
#             'length_truncate': lambda q, i: self.length_perturbation(q, 'truncate'),
#             'length_expand': lambda q, i: self.length_perturbation(q, 'expand'),
#             'word_shuffle': lambda q, i: self.word_order_shuffle(q, i),
#             'noise': lambda q, i: self.noise_injection(q),
#             'missing': lambda q, i: self.missing_terms(q, i),
#             'encoding': lambda q, i: self.encoding_corruption(q)
#         }
        
#         if pert_type in perturbation_map:
#             return perturbation_map[pert_type](query, intensity)
#         else:
#             return query
    
#     def _calculate_result_similarity(self, orig_candidates: List[str], 
#                                    pert_candidates: List[str],
#                                    orig_scores: List[float], 
#                                    pert_scores: List[float]) -> float:
#         """Calculate similarity between two result sets"""
#         # Jaccard similarity for candidates
#         orig_set = set(orig_candidates)
#         pert_set = set(pert_candidates)
#         jaccard = len(orig_set & pert_set) / len(orig_set | pert_set) if orig_set | pert_set else 0
        
#         # Score correlation for common candidates
#         score_corr = 0
#         if orig_set & pert_set:
#             common_orig_scores = []
#             common_pert_scores = []
#             for i, cand in enumerate(orig_candidates):
#                 if cand in pert_candidates:
#                     j = pert_candidates.index(cand)
#                     common_orig_scores.append(orig_scores[i])
#                     common_pert_scores.append(pert_scores[j])
            
#             if len(common_orig_scores) > 1:
#                 score_corr, _ = pearsonr(common_orig_scores, common_pert_scores)
#                 score_corr = max(0, score_corr)  # Only positive correlation matters
        
#         return (jaccard + score_corr) / 2
    
#     def _calculate_rank_correlation(self, orig_candidates: List[str], 
#                                   pert_candidates: List[str],
#                                   orig_scores: List[float], 
#                                   pert_scores: List[float]) -> float:
#         """Calculate rank correlation between result sets"""
#         common_candidates = list(set(orig_candidates) & set(pert_candidates))
#         if len(common_candidates) < 2:
#             return 0.0
        
#         orig_ranks = []
#         pert_ranks = []
        
#         for cand in common_candidates:
#             orig_ranks.append(orig_candidates.index(cand))
#             pert_ranks.append(pert_candidates.index(cand))
        
#         corr, _ = spearmanr(orig_ranks, pert_ranks)
#         return corr if not np.isnan(corr) else 0.0
    
#     def _get_bm25_scores(self, query: str, candidates: List[str]) -> List[float]:
#         """Placeholder for BM25 scoring - implement with your BM25 model"""
#         # This should interface with your actual BM25 implementation
#         return [random.random() for _ in candidates]
    
#     def _get_embedding(self, query: str, model) -> np.ndarray:
#         """Get embedding from model - implement with your embedding model"""
#         # This should interface with your actual embedding model
#         return np.random.rand(384)  # Placeholder
    
#     def _get_cross_encoder_scores(self, query: str, candidates: List[str]) -> List[float]:
#         """Get cross-encoder scores - implement with your cross-encoder"""
#         # This should interface with your actual cross-encoder
#         return [random.random() for _ in candidates]
    
#     def _deduplicate_candidates(self, candidates: List[str]) -> List[str]:
#         """Deduplicate candidates - implement your deduplication logic"""
#         # This should interface with your actual deduplication logic
#         seen = set()
#         result = []
#         for cand in candidates:
#             normalized = cand.lower().strip()
#             if normalized not in seen:
#                 seen.add(normalized)
#                 result.append(cand)
#         return result
    







import numpy as np
import random
import string
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

class PerturbationType(Enum):
    """Types of character-level perturbations"""
    TYPO = "typo"
    DELETION = "deletion"
    INSERTION = "insertion"
    SUBSTITUTION = "substitution"
    TRANSPOSITION = "transposition"
    DUPLICATION = "duplication"
    CASE_CHANGE = "case_change"
    UNICODE_SUBSTITUTION = "unicode_substitution"
    WHITESPACE_NOISE = "whitespace_noise"
    PUNCTUATION_NOISE = "punctuation_noise"

@dataclass
class TestCase:
    """Represents a single test case"""
    original_query: str
    perturbed_query: str
    perturbation_type: PerturbationType
    intensity: float
    expected_degradation: float

class CharacterPerturbator:
    """Handles character-level perturbations with adjustable intensity"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common typo mappings
        self.typo_map = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'n', 'g'],
            'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'c'],
            'e': ['w', 'r', '3'],
            'f': ['d', 'g', 'r'],
            'g': ['f', 'h', 't'],
            'h': ['g', 'j', 'y'],
            'i': ['u', 'o', '8'],
            'j': ['h', 'k', 'u'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'p', '1'],
            'm': ['n', 'j'],
            'n': ['b', 'm', 'h'],
            'o': ['i', 'p', '0'],
            'p': ['o', 'l', '0'],
            'q': ['w', 'a', '1'],
            'r': ['e', 't', '4'],
            's': ['a', 'd', 'z'],
            't': ['r', 'y', '5'],
            'u': ['y', 'i', '7'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', '2'],
            'x': ['z', 'c', 's'],
            'y': ['t', 'u', '6'],
            'z': ['a', 'x', 's']
        }
        
        # Unicode lookalikes
        self.unicode_map = {
            'a': ['а', 'ɑ', 'α'],
            'e': ['е', 'ε', 'ɛ'],
            'i': ['і', 'ι', 'ɪ'],
            'o': ['о', 'ο', 'σ'],
            'c': ['с', 'ϲ'],
            'p': ['р', 'ρ'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
        }
    
    def apply_perturbation(self, text: str, perturbation_type: PerturbationType, 
                          intensity: float) -> str:
        """Apply a specific perturbation type with given intensity"""
        if perturbation_type == PerturbationType.TYPO:
            return self._apply_typo(text, intensity)
        elif perturbation_type == PerturbationType.DELETION:
            return self._apply_deletion(text, intensity)
        elif perturbation_type == PerturbationType.INSERTION:
            return self._apply_insertion(text, intensity)
        elif perturbation_type == PerturbationType.SUBSTITUTION:
            return self._apply_substitution(text, intensity)
        elif perturbation_type == PerturbationType.TRANSPOSITION:
            return self._apply_transposition(text, intensity)
        elif perturbation_type == PerturbationType.DUPLICATION:
            return self._apply_duplication(text, intensity)
        elif perturbation_type == PerturbationType.CASE_CHANGE:
            return self._apply_case_change(text, intensity)
        elif perturbation_type == PerturbationType.UNICODE_SUBSTITUTION:
            return self._apply_unicode_substitution(text, intensity)
        elif perturbation_type == PerturbationType.WHITESPACE_NOISE:
            return self._apply_whitespace_noise(text, intensity)
        elif perturbation_type == PerturbationType.PUNCTUATION_NOISE:
            return self._apply_punctuation_noise(text, intensity)
        else:
            return text
    
    def _apply_typo(self, text: str, intensity: float) -> str:
        """Simulate keyboard typos"""
        chars = list(text)
        num_changes = max(1, int(len(chars) * intensity))
        
        for _ in range(num_changes):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx].lower()
            
            if char in self.typo_map:
                chars[idx] = random.choice(self.typo_map[char])
                if text[idx].isupper():
                    chars[idx] = chars[idx].upper()
        
        return ''.join(chars)
    
    def _apply_deletion(self, text: str, intensity: float) -> str:
        """Delete random non-space characters"""
        chars = list(text)
        num_deletions = max(1, int(len(chars) * intensity))
        
        for _ in range(num_deletions):
            # Find indices of non-space characters
            non_space_indices = [i for i, c in enumerate(chars) if c != ' ']
            if not non_space_indices:
                break
            idx = random.choice(non_space_indices)
            del chars[idx]
        
        return ''.join(chars)
    
    # def _apply_insertion(self, text: str, intensity: float) -> str:
    #     """Insert random characters"""
    #     chars = list(text)
    #     num_insertions = max(1, int(len(chars) * intensity))
        
    #     for _ in range(num_insertions):
    #         idx = random.randint(0, len(chars))
    #         char = random.choice(string.ascii_lowercase)
    #         chars.insert(idx, char)
        
    #     return ''.join(chars)
    


    def _apply_insertion(self, text: str, intensity: float) -> str:
        """Insert realistic characters (adjacent on keyboard)"""
        chars = list(text)
        num_insertions = max(1, int(len(chars) * intensity))
        for _ in range(num_insertions):
            idx = random.randint(0, len(chars))
            # Insert a nearby key if possible
            if idx > 0 and chars[idx-1].lower() in self.typo_map:
                if random.random() < 0.5:
                    char = chars[idx-1]
                else:
                    char = random.choice(self.typo_map[chars[idx-1].lower()])
            else:
                char = random.choice(string.ascii_lowercase)
            chars.insert(idx, char)
        return ''.join(chars)

    def _apply_substitution(self, text: str, intensity: float) -> str:
        """Substitute random characters"""
        chars = list(text)
        num_substitutions = max(1, int(len(chars) * intensity))
        
        for _ in range(num_substitutions):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice(string.ascii_letters)
        
        return ''.join(chars)
    
    def _apply_transposition(self, text: str, intensity: float) -> str:
        """Transpose adjacent characters"""
        chars = list(text)
        num_transpositions = max(1, int(len(chars) * intensity))
        
        for _ in range(num_transpositions):
            if len(chars) <= 1:
                break
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        
        return ''.join(chars)
    
    def _apply_duplication(self, text: str, intensity: float) -> str:
        """Duplicate random characters"""
        chars = list(text)
        num_duplications = max(1, int(len(chars) * intensity))
        
        for _ in range(num_duplications):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
        
        return ''.join(chars)
    
    def _apply_case_change(self, text: str, intensity: float) -> str:
        """Randomly change case of characters"""
        chars = list(text)
        num_changes = max(1, int(len(chars) * intensity))
        
        for _ in range(num_changes):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                chars[idx] = chars[idx].swapcase()
        
        return ''.join(chars)
    
    def _apply_unicode_substitution(self, text: str, intensity: float) -> str:
        """Replace characters with unicode lookalikes"""
        chars = list(text)
        num_substitutions = max(1, int(len(chars) * intensity))
        
        for _ in range(num_substitutions):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx].lower()
            
            if char in self.unicode_map:
                chars[idx] = random.choice(self.unicode_map[char])
        
        return ''.join(chars)
    
    def _apply_whitespace_noise(self, text: str, intensity: float) -> str:
        """Add or remove whitespace"""
        words = text.split()
        
        if random.random() < 0.5:
            # Add extra spaces
            num_additions = max(1, int(len(words) * intensity))
            for _ in range(num_additions):
                if not words:
                    break
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx] + ' '
        else:
            # Remove spaces
            if len(words) > 1:
                num_merges = max(1, int((len(words) - 1) * intensity))
                for _ in range(num_merges):
                    if len(words) <= 1:
                        break
                    idx = random.randint(0, len(words) - 2)
                    words[idx] = words[idx] + words[idx + 1]
                    del words[idx + 1]
        
        return ' '.join(words)
    
    def _apply_punctuation_noise(self, text: str, intensity: float) -> str:
        """Add, remove, or change punctuation"""
        chars = list(text)
        punctuation = string.punctuation
        num_changes = max(1, int(len(chars) * intensity))
        
        for _ in range(num_changes):
            action = random.choice(['add', 'remove', 'change'])
            
            if action == 'add':
                idx = random.randint(0, len(chars))
                chars.insert(idx, random.choice(punctuation))
            elif action == 'remove':
                punct_indices = [i for i, c in enumerate(chars) if c in punctuation]
                if punct_indices:
                    idx = random.choice(punct_indices)
                    del chars[idx]
            elif action == 'change':
                punct_indices = [i for i, c in enumerate(chars) if c in punctuation]
                if punct_indices:
                    idx = random.choice(punct_indices)
                    chars[idx] = random.choice(punctuation)
        
        return ''.join(chars)

class HybridIRModelTester:
    """Main testing framework for hybrid IR models"""
    
    def __init__(self, model_interface: Callable):
        """
        Initialize tester with model interface
        
        Args:
            model_interface: Function that takes query and returns ranked results
                            Should return List[Tuple[doc_id, score]]
        """
        self.model = model_interface
        self.perturbator = CharacterPerturbator()
        self.test_results = defaultdict(list)
    
    def generate_test_cases(self, queries: List[str], 
                           perturbation_types: List[PerturbationType] = None,
                           intensity_levels: List[float] = None) -> List[TestCase]:
        """Generate test cases with various perturbations"""
        if perturbation_types is None:
            perturbation_types = list(PerturbationType)
        
        if intensity_levels is None:
            intensity_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        test_cases = []
        
        for query in queries:
            for p_type, intensity_levels in perturbation_types.items():
                for intensity in intensity_levels:
                    perturbed = self.perturbator.apply_perturbation(
                        query, p_type, intensity
                    )
                    
                    # Expected degradation is proportional to intensity
                    # This is a simple heuristic; adjust based on your model
                    expected_degradation = intensity * 0.5
                    
                    test_cases.append(TestCase(
                        original_query=query,
                        perturbed_query=perturbed,
                        perturbation_type=p_type,
                        intensity=intensity,
                        expected_degradation=expected_degradation
                    ))
        
        return test_cases
    
    def run_sensitivity_test(self, test_cases: List[TestCase]) -> Dict:
        """Run sensitivity tests and collect results"""
        results = {
            'summary': {},
            'details': [],
            'by_perturbation': defaultdict(list),
            'by_intensity': defaultdict(list)
        }
        
        for test_case in test_cases:
            # Get results for original and perturbed queries
            original_results = self.model(test_case.original_query)
            perturbed_results = self.model(test_case.perturbed_query)
            
            # Calculate metrics
            metrics = self._calculate_metrics(original_results, perturbed_results)
            
            # Store results
            result = {
                'test_case': test_case,
                'metrics': metrics,
                'degradation': metrics['rank_correlation_degradation']
            }
            
            results['details'].append(result)
            results['by_perturbation'][test_case.perturbation_type].append(result)
            results['by_intensity'][test_case.intensity].append(result)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary(results)
        
        return results
    
    def _calculate_metrics(self, original: List[Tuple], perturbed: List[Tuple]) -> Dict:
        """Calculate various metrics comparing original and perturbed results"""
        metrics = {}
        
        # Extract document IDs
        orig_docs = [doc_id for doc_id, _ in original]
        pert_docs = [doc_id for doc_id, _ in perturbed]
        
        # Overlap metrics
        common_docs = set(orig_docs) & set(pert_docs)
        metrics['overlap_ratio'] = len(common_docs) / len(orig_docs) if orig_docs else 0
        
        # Rank correlation (Kendall's tau approximation)
        if common_docs:
            orig_ranks = {doc: i for i, doc in enumerate(orig_docs)}
            pert_ranks = {doc: i for i, doc in enumerate(pert_docs)}
            
            rank_diffs = []
            for doc in common_docs:
                if doc in orig_ranks and doc in pert_ranks:
                    rank_diffs.append(abs(orig_ranks[doc] - pert_ranks[doc]))
            
            metrics['avg_rank_diff'] = np.mean(rank_diffs) if rank_diffs else 0
            metrics['max_rank_diff'] = max(rank_diffs) if rank_diffs else 0
            
            # Normalized rank correlation (1 = perfect, 0 = no correlation)
            max_possible_diff = len(orig_docs) - 1
            metrics['rank_correlation'] = 1 - (metrics['avg_rank_diff'] / max_possible_diff)
            metrics['rank_correlation_degradation'] = 1 - metrics['rank_correlation']
        else:
            metrics['avg_rank_diff'] = len(orig_docs)
            metrics['max_rank_diff'] = len(orig_docs)
            metrics['rank_correlation'] = 0
            metrics['rank_correlation_degradation'] = 1
        
        # Score metrics
        orig_scores = {doc_id: score for doc_id, score in original}
        pert_scores = {doc_id: score for doc_id, score in perturbed}
        
        if common_docs:
            score_diffs = []
            for doc in common_docs:
                if doc in orig_scores and doc in pert_scores:
                    orig_score = orig_scores[doc]
                    pert_score = pert_scores[doc]
                    if orig_score > 0:
                        score_diffs.append(abs(orig_score - pert_score) / orig_score)
            
            metrics['avg_score_diff_ratio'] = np.mean(score_diffs) if score_diffs else 0
        else:
            metrics['avg_score_diff_ratio'] = 1
        
        return metrics
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        summary = {}
        
        # Overall statistics
        all_degradations = [r['degradation'] for r in results['details']]
        summary['avg_degradation'] = np.mean(all_degradations)
        summary['std_degradation'] = np.std(all_degradations)
        summary['max_degradation'] = max(all_degradations)
        summary['min_degradation'] = min(all_degradations)
        
        # By perturbation type
        summary['by_perturbation'] = {}
        for p_type, results_list in results['by_perturbation'].items():
            degradations = [r['degradation'] for r in results_list]
            summary['by_perturbation'][p_type.value] = {
                'avg': np.mean(degradations),
                'std': np.std(degradations),
                'count': len(degradations)
            }
        
        # By intensity
        summary['by_intensity'] = {}
        for intensity, results_list in results['by_intensity'].items():
            degradations = [r['degradation'] for r in results_list]
            summary['by_intensity'][intensity] = {
                'avg': np.mean(degradations),
                'std': np.std(degradations),
                'count': len(degradations)
            }
        
        return summary
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable report"""
        report = []
        report.append("=== Hybrid IR Model Sensitivity Test Report ===\n")
        
        # Overall summary
        summary = results['summary']
        report.append(f"Overall Performance:")
        report.append(f"  Average Degradation: {summary['avg_degradation']:.3f}")
        report.append(f"  Std Dev: {summary['std_degradation']:.3f}")
        report.append(f"  Range: [{summary['min_degradation']:.3f}, {summary['max_degradation']:.3f}]")
        report.append("")
        
        # By perturbation type
        report.append("Performance by Perturbation Type:")
        for p_type, stats in summary['by_perturbation'].items():
            report.append(f"  {p_type}:")
            report.append(f"    Avg Degradation: {stats['avg']:.3f} (±{stats['std']:.3f})")
            report.append(f"    Test Count: {stats['count']}")
        report.append("")
        
        # By intensity
        report.append("Performance by Intensity Level:")
        for intensity, stats in sorted(summary['by_intensity'].items()):
            report.append(f"  {intensity:.0%} intensity:")
            report.append(f"    Avg Degradation: {stats['avg']:.3f} (±{stats['std']:.3f})")
        report.append("")
        
        # Robustness assessment
        report.append("Robustness Assessment:")
        avg_deg = summary['avg_degradation']
        if avg_deg < 0.1:
            report.append("  ✓ Excellent: Model is highly robust to character-level noise")
        elif avg_deg < 0.2:
            report.append("  ✓ Good: Model handles most perturbations well")
        elif avg_deg < 0.3:
            report.append("  ⚠ Fair: Model shows moderate sensitivity to perturbations")
        else:
            report.append("  ✗ Poor: Model is highly sensitive to character-level noise")
        
        # Identify weaknesses
        report.append("\nIdentified Weaknesses:")
        weak_types = [(p_type, stats) for p_type, stats in summary['by_perturbation'].items() 
                      if stats['avg'] > summary['avg_degradation'] * 1.2]
        
        if weak_types:
            for p_type, stats in sorted(weak_types, key=lambda x: x[1]['avg'], reverse=True):
                report.append(f"  - {p_type}: {stats['avg']:.3f} degradation (above average)")
        else:
            report.append("  - No significant weaknesses identified")
        
        return "\n".join(report)

