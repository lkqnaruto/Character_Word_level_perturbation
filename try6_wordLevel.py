import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Set
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
import string
import re
import random
from typing import Tuple, Dict, List, Optional
from nltk.corpus import wordnet


import re, random, functools
from typing import Tuple, Dict, List, Optional
from nltk.corpus import wordnet
from nltk.corpus import words as nltk_words






@functools.lru_cache(maxsize=8192)
def _synonyms_for(token: str, pos=None):
    syns = []
    synsets = wordnet.synsets(token, pos=pos) if pos else wordnet.synsets(token)
    for syn in synsets:
        for lemma in syn.lemmas():
            cand = lemma.name().replace('_', ' ')
            if cand.lower() != token:
                syns.append(cand)
    # keep short/common-looking single words first
    syns = [s for s in syns if s.isalpha() and 2 < len(s) < 30]
    return syns


# --- helpers 
_WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"          # handles hyphenated words & apostrophes: policy-maker, bank's
_PLACEHOLDER = r"__PHRASE_\d+__"
_PUNCT = r"[^\w\s]"                             # any single non-word, non-space (.,;:!?()[]{}"”’ etc.)
_NUMBER = r"\d+(?:[.,]\d+)*%?|\$\d+(?:[.,]\d+)*"
TOKEN_RX = re.compile(fr"{_PLACEHOLDER}|{_WORD}|{_PUNCT}|{_NUMBER}")

def tokenize(text: str):
    # Returns a list of tokens: words/placeholders/punctuation
    return TOKEN_RX.findall(text)

def detokenize(tokens):
    out = []
    for t in tokens:
        if not out:
            out.append(t)
            continue
        # If current token is punctuation, attach to previous without space
        if re.fullmatch(_PUNCT, t):
            out[-1] += t
        else:
            # otherwise add a space then the token
            out.append(" " + t)
    return "".join(out)




import re
import random
from typing import Tuple, Dict, List, Optional


# Download required NLTK data (run once)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

class WordPerturbationType(Enum):
    """Types of word-level perturbations"""
    SYNONYM_REPLACEMENT = "synonym_replacement"
    WORD_DELETION = "word_deletion"
    WORD_INSERTION = "word_insertion"
    WORD_REORDERING = "word_reordering"
    WORD_DUPLICATION = "word_duplication"
    STOP_WORD_REMOVAL = "stop_word_removal"
    STEMMING_VARIATION = "stemming_variation"
    HYPONYM_REPLACEMENT = "hyponym_replacement"
    HYPERNYM_REPLACEMENT = "hypernym_replacement"
    RELATED_WORD_REPLACEMENT = "related_word_replacement"
    RANDOM_WORD_REPLACEMENT = "random_word_replacement"
    PHRASE_PARAPHRASING = "phrase_paraphrasing"
    WORD_SPLITTING = "word_splitting"
    WORD_MERGING = "word_merging"
    ABBREVIATION = "abbreviation"

@dataclass
class WordTestCase:
    """Represents a word-level test case"""
    original_query: str
    perturbed_query: str
    perturbation_type: WordPerturbationType
    intensity: float
    affected_words: List[str]
    perturbation_details: Dict

class WordPerturbator:
    """Handles word-level perturbations with adjustable intensity"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common stop words
        self.stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'of', 'in',
            'to', 'for', 'with', 'as', 'by', 'that', 'this', 'it', 'from',
            'or', 'but', 'not', 'are', 'was', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'could', 'be', 'being', 'been'
        }
        
        # Common abbreviations
        self.abbreviations = {
            'information': ['info', 'inf'],
            'retrieval': ['ret', 'retr'],
            'system': ['sys', 'syst'],
            'algorithm': ['algo', 'alg'],
            'machine': ['mach', 'mc'],
            'learning': ['learn', 'lrn'],
            'processing': ['proc', 'process'],
            'natural': ['nat', 'natl'],
            'language': ['lang', 'lng'],
            'neural': ['neur', 'nn'],
            'network': ['net', 'ntwk'],
            'database': ['db', 'dbase'],
            'application': ['app', 'appl'],
            'development': ['dev', 'devel'],
            'management': ['mgmt', 'mgt', 'mgnt']
        }
        
        # Common semantic replacements
        self.semantic_replacements = {
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'minute', 'compact'],
            'good': ['great', 'excellent', 'fine', 'superior'],
            'bad': ['poor', 'terrible', 'awful', 'inferior'],
            'search': ['find', 'lookup', 'query', 'seek'],
            'retrieve': ['fetch', 'get', 'obtain', 'access'],
            'model': ['system', 'framework', 'approach', 'method'],
            'data': ['information', 'content', 'records', 'dataset']
        }
        
        # Word insertion vocabulary (common filler words)
        self.insertion_vocab = [
            "very","quite","really","actually","particularly","especially",
            "significantly","notably","specific","certain","various","relevant",
            "appropriate","applicable","designated","related","important","main",
            "key","primary","critical","essential","necessary","mandatory",
            "generally","typically","normally","commonly","regularly","usually",
            "periodically","routinely"
        ]
    
        self.default_exclude_phrases = [
            # Regulatory / Compliance
            "Anti Money Laundering",
            "Know Your Customer",
            "Customer Due Diligence",
            "Suspicious Activity Report",
            "Office of Foreign Assets Control",
            "Bank Secrecy Act",
            "Fair Lending Practices",
            "Truth in Lending Act",
            "Community Reinvestment Act",

            # Risk & Governance
            "Model Risk Management",
            "Enterprise Risk Management",
            "Operational Risk Framework",
            "Three Lines of Defense",
            "Stress Testing",
            "Capital Adequacy Ratio",
            "Basel III",
            "Basel IV",

            # Financial Products & Services
            "Credit Card Agreement",
            "Home Equity Line of Credit",
            "Certificate of Deposit",
            "Treasury Bill",
            "Mortgage Backed Securities",
            "Overdraft Protection",
            "Foreign Exchange Transaction",
            "Wire Transfer",
            "Payment Card Industry Data Security Standard",

            # Internal Procedures
            "Standard Operating Procedure",
            "Internal Audit",
            "Risk Assessment Report",
            "Code of Conduct",
            "Conflict of Interest Policy",

            "machine learning", "deep learning", "natural language processing", "neural network",
        ]

        self.default_exclude_words = ["AI", "ML", "DL", "GenAI", "policy"
                                      # Regulatory bodies
                                    "OFAC", "SEC", "FDIC", "FINRA", "OCC", "FRB", "CFPB",

                                    # Common finance abbreviations
                                    "APR", "APY", "LIBOR", "SOFR", "KYC", "AML", "SAR", "FATCA",

                                    # Financial instruments / terms
                                    "Bond", "Stock", "ETF", "Loan", "Deposit", "Derivative",
                                    "Mortgage", "Dividend", "Collateral", "Liquidity",

                                    # Risk terms
                                    "VaR", "Stress", "Credit", "Market", "Operational", "Capital"
                                ]


    def apply_perturbation(self, text: str, perturbation_type: WordPerturbationType, 
                          intensity: float) -> Tuple[str, Dict]:
        """Apply a specific word-level perturbation with given intensity"""
        
        perturbation_methods = {
            WordPerturbationType.SYNONYM_REPLACEMENT: self._apply_synonym_replacement,
            WordPerturbationType.WORD_DELETION: self._apply_word_deletion,
            WordPerturbationType.WORD_INSERTION: self._apply_word_insertion,
            WordPerturbationType.WORD_REORDERING: self._apply_word_reordering,
            # WordPerturbationType.WORD_DUPLICATION: self._apply_word_duplication,
            # WordPerturbationType.STOP_WORD_REMOVAL: self._apply_stop_word_removal,
            # WordPerturbationType.STEMMING_VARIATION: self._apply_stemming_variation,
            # WordPerturbationType.HYPONYM_REPLACEMENT: self._apply_hyponym_replacement,
            # WordPerturbationType.HYPERNYM_REPLACEMENT: self._apply_hypernym_replacement,
            # WordPerturbationType.RELATED_WORD_REPLACEMENT: self._apply_related_word_replacement,
            WordPerturbationType.RANDOM_WORD_REPLACEMENT: self._apply_random_word_replacement,
            # WordPerturbationType.PHRASE_PARAPHRASING: self._apply_phrase_paraphrasing,
            WordPerturbationType.WORD_SPLITTING: self._apply_word_splitting,
            WordPerturbationType.WORD_MERGING: self._apply_word_merging,
            # WordPerturbationType.ABBREVIATION: self._apply_abbreviation
        }
        
        method = perturbation_methods.get(perturbation_type)
        if method:
            return method(text, intensity)
        else:
            return text, {}
    
    # def _get_wordnet_pos(self, word: str) -> str:
    #     """Get WordNet POS tag"""
    #     # Simplified POS tagging
    #     if word.endswith('ing'):
    #         return wordnet.VERB
    #     elif word.endswith('ly'):
    #         return wordnet.ADV
    #     elif word.endswith('ed'):
    #         return wordnet.VERB
    #     else:
    #         return wordnet.NOUN

    def _nltk_pos_to_wordnet(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def _get_wordnet_pos(self, word: str):
        tag = nltk.pos_tag([word])[0][1]
        return self._nltk_pos_to_wordnet(tag)


    # def _apply_synonym_replacement(
    #     self,
    #     text: str,
    #     intensity: float,
    #     exclude_phrases: Optional[List[str]] = None,   # NEW: multi-word exclusions
    #     exclude_words: Optional[List[str]] = None      # NEW: single-word exclusions
    # ) -> Tuple[str, Dict]:
    #     """Replace words with synonyms, honoring phrase/word exclusion lists."""

    #     exclude_phrases = exclude_phrases if exclude_phrases is not None else self.default_exclude_phrases
    #     exclude_words = exclude_words if exclude_words is not None else self.default_exclude_words

    #     # print(f"Excluding phrases: {exclude_phrases}, words: {exclude_words}")
    #     # --- 1) Protect excluded phrases with placeholders (case-insensitive) ---
    #     # Sort longest first to avoid partial overlaps
    #     phrases_sorted = sorted(exclude_phrases, key=len, reverse=True)
    #     protected_map = {}
    #     protected_text = text
    #     for i, phrase in enumerate(phrases_sorted):
    #         if not phrase.strip():
    #             continue
    #         placeholder = f"__PHRASE_{i}__"
    #         # word-boundary-ish match but allow spaces within phrase
    #         pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    #         protected_text, n_subs = pattern.subn(placeholder, protected_text)
    #         if n_subs:
    #             protected_map[placeholder] = phrase  # remember original

    #     # --- 2) Tokenize simply by spaces (your original behavior) ---
    #     words = protected_text.split()

    #     # Compute how many to attempt
    #     num_replacements = max(1, int(len(words) * intensity))
    #     affected_words = []
    #     replacements = {}

    #     # --- 3) Build candidate list (exclude stop words, short tokens, placeholders, and exclude_words) ---
    #     def is_placeholder(w: str) -> bool:
    #         return w in protected_map  # exact token match

    #     content_words = [
    #         (i, w) for i, w in enumerate(words)
    #         if (len(w) > 2)
    #         and (w.lower() not in getattr(self, "stop_words", set()))
    #         and (w.lower() not in exclude_words)
    #         and (not is_placeholder(w))
    #         and w.isalpha()  # avoid numbers/mixed tokens; keep simple
    #     ]

    #     if content_words:
    #         # --- 4) Your hybrid selection logic (kept) ---
    #         if len(content_words) < 6:
    #             # Probabilistic selection for short texts
    #             selected = [cw for cw in content_words if random.random() < intensity]
    #             # ensure at least one if intensity was tiny
    #             if not selected:
    #                 selected = random.sample(content_words, 1)
    #         else:
    #             # Deterministic selection for longer texts
    #             num_replacements = max(1, int(len(content_words) * intensity))
    #             selected = random.sample(content_words, min(num_replacements, len(content_words)))

    #         # --- 5) Replace with WordNet synonyms (caps preserved) ---
    #         for idx, word in selected:
    #             # Skip if it turned into a placeholder after earlier edits
    #             if is_placeholder(words[idx]):
    #                 continue

    #             lower = word.lower()
    #             syns = []
    #             for syn in wordnet.synsets(lower):
    #                 for lemma in syn.lemmas():
    #                     cand = lemma.name().replace('_', ' ')
    #                     if cand.lower() != lower:
    #                         syns.append(cand)

    #             if syns:
    #                 # Trim the candidate set a bit for stability, then choose
    #                 choices = syns[:5] if len(syns) > 5 else syns
    #                 replacement = random.choice(choices)

    #                 # Preserve capitalization of the first letter
    #                 if word[0].isupper():
    #                     replacement = replacement.capitalize()

    #                 affected_words.append(word)
    #                 replacements.setdefault(word, replacement)  # keeps last picked if duplicates
    #                 words[idx] = replacement

    #     # --- 6) Reconstruct and restore phrases ---
    #     replaced_text = ' '.join(words)
    #     for placeholder, phrase in protected_map.items():
    #         replaced_text = replaced_text.replace(placeholder, phrase)

    #     return replaced_text, {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }

    def _apply_synonym_replacement(
        self,
        text: str,
        intensity: float,
        exclude_phrases: Optional[List[str]] = None,
        exclude_words: Optional[List[str]] = None
    ) -> Tuple[str, Dict]:

        # --- exclusions (normalize words to lowercase set) ---
        exclude_phrases = exclude_phrases if exclude_phrases is not None else self.default_exclude_phrases
        exclude_words = exclude_words if exclude_words is not None else self.default_exclude_words
        exclude_words = set(w.lower() for w in (exclude_words or []))
        stop_words = getattr(self, "stop_words", set())
        stop_words = set(w.lower() for w in stop_words)

        # --- protect phrases (case-insensitive), store matched text to preserve case ---
        phrases_sorted = sorted(exclude_phrases or [], key=len, reverse=True)
        protected_map = {}
        protected_text = text
        for i, phrase in enumerate(phrases_sorted):
            if not phrase.strip():
                continue
            placeholder = f"__PHRASE_{i}__"

            def _sub(m):
                protected_map[placeholder] = m.group(0)  # store matched form w/ original case
                return placeholder

            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            protected_text = pattern.sub(_sub, protected_text)
        # print(protected_text)
        # --- tokenization compatible with your original join ---
        words = protected_text.split()

        affected_words: List[str] = []
        replacements: Dict[str, str] = {}

        # candidates: ignore placeholders, stopwords, excluded, short tokens, non-alpha
        def is_placeholder(w: str) -> bool:
            return w in protected_map

        content_words = [
            (i, w) for i, w in enumerate(words)
            if (len(w) > 2)
            and w.isalpha()
            and (w.lower() not in stop_words)
            and (w.lower() not in exclude_words)
            and (not is_placeholder(w))
        ]

        selected = []
        if content_words and intensity > 0.0:
            n = len(content_words)
            # Deterministic-by-count even for short texts
            k = int(round(n * intensity))
            k = max(0, min(k, n))
            if k > 0:
                selected = random.sample(content_words, k)


        for idx, word in selected:
            if is_placeholder(words[idx]):
                continue
            lower = word.lower()

            pos = self._get_wordnet_pos(lower)  # None by default; plug in tagger if available
            syns = _synonyms_for(lower, pos=pos)

            if syns:
                # prefer early/common senses; cap for stability
                choices = syns[:5] if len(syns) > 5 else syns
                replacement = random.choice(choices)

                if word[0].isupper():
                    replacement = replacement.capitalize()

                affected_words.append(word)
                replacements[word] = replacement  # last wins (or switch to index map)
                words[idx] = replacement

        # --- restore phrases with original matched casing ---
        replaced_text = ' '.join(words)
        for placeholder, matched in protected_map.items():
            replaced_text = replaced_text.replace(placeholder, matched)

        return replaced_text, {
            "affected_words": affected_words,
            "replacements": replacements,
            "num_replacements": len(affected_words),
        }




    # def _apply_synonym_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Replace words with synonyms"""
    #     words = text.split()
    #     num_replacements = max(1, int(len(words) * intensity))
    #     affected_words = []
    #     replacements = {}
        
    #     # Select words to replace (excluding stop words)
    #     content_words = [(i, w) for i, w in enumerate(words) 
    #                     if w.lower() not in self.stop_words and len(w) > 2]
        
    #     if content_words:

    #         # Hybrid selection logic
    #         if len(content_words) < 6:
    #             # Probabilistic selection for short texts
    #             selected = [cw for cw in content_words if random.random() < intensity]
    #         else:
    #             # Deterministic selection for longer texts
    #             num_replacements = max(1, int(len(content_words) * intensity))
    #             selected = random.sample(content_words, min(num_replacements, len(content_words)))

    #         # selected = random.sample(content_words, 
    #         #                        min(num_replacements, len(content_words)))
            
            
    #         for idx, word in selected:
    #             # Try WordNet synonyms first
    #             synonyms = []
    #             for syn in wordnet.synsets(word.lower()):
    #                 for lemma in syn.lemmas():
    #                     synonym = lemma.name().replace('_', ' ')
    #                     if synonym.lower() != word.lower():
    #                         synonyms.append(synonym)
                
    #             # Fallback to predefined synonyms
    #             # if not synonyms and word.lower() in self.semantic_replacements:
    #             #     synonyms = self.semantic_replacements[word.lower()]
    #             print(word, synonyms)
    #             if synonyms:
    #                 replacement = random.choice(synonyms[:5])  # Limit choices
    #                 # Preserve capitalization
    #                 if word[0].isupper():
    #                     replacement = replacement.capitalize()
                    
    #                 affected_words.append(word)
    #                 replacements[word] = replacement
    #                 words[idx] = replacement
        
    #     return ' '.join(words), {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }
    
    def _apply_word_deletion(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Delete random words"""
        # words = text.split()
        words = tokenize(text)
        if len(words) <= 1:
            return text, {'affected_words': [], 'num_deletions': 0}
        
        num_deletions = max(1, int(len(words) * intensity))
        num_deletions = min(num_deletions, len(words) - 1)  # Keep at least one word
        
        # Prefer deleting non-essential words first
        deletion_candidates = []
        for i, word in enumerate(words):
            if word.lower() in self.stop_words:
                deletion_candidates.append((i, word, 0))  # Priority 0 for stop words
            else:
                deletion_candidates.append((i, word, 1))  # Priority 1 for content words

        # Sort by priority (delete stop words first)
        deletion_candidates.sort(key=lambda x: x[2])

        indices_to_delete = set()
        affected_words = []
        
        for idx, word, _ in deletion_candidates[:num_deletions]:
            indices_to_delete.add(idx)
            affected_words.append(word)
        
        remaining_words = [w for i, w in enumerate(words) if i not in indices_to_delete]
        detokenized_remaining = detokenize(remaining_words)
        return detokenized_remaining, {
            'affected_words': affected_words,
            'deleted_positions': list(indices_to_delete),
            'num_deletions': len(affected_words)
        }
    
    def _apply_word_insertion(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Insert random words"""
        words = tokenize(text)
        num_insertions = max(1, int(len(words) * intensity))
        
        inserted_words = []
        insertion_positions = []
        
        for _ in range(num_insertions):
            position = random.randint(0, len(words))
            word_to_insert = random.choice(self.insertion_vocab)
            words.insert(position, word_to_insert)
            inserted_words.append(word_to_insert)
            insertion_positions.append(position)
        
        return detokenize(words), {
            'inserted_words': inserted_words,
            'insertion_positions': insertion_positions,
            'num_insertions': len(inserted_words)
        }
    

 

    # Assumes you already have these; if not, use the versions I shared earlier
    # tokenize(text)  -> splits into placeholders/words/punct tokens
    # detokenize(tokens) -> joins while keeping punctuation tight

    # def _apply_word_insertion(
    #     self,
    #     text: str,
    #     intensity: float,
    #     exclude_phrases: Optional[List[str]] = None,   # multi-word phrases to protect
    #     exclude_words: Optional[List[str]] = None      # words you never want to insert
    # ) -> Tuple[str, Dict]:
    #     """Insert words from self.insertion_vocab, honoring phrase/word exclusions and punctuation."""

    #     # --- config & normalization ---
    #     intensity = max(0.0, min(1.0, float(intensity)))
    #     exclude_phrases = exclude_phrases if exclude_phrases is not None else getattr(self, "default_exclude_phrases", []) or []
    #     exclude_words = set(w.lower() for w in (exclude_words if exclude_words is not None else getattr(self, "default_exclude_words", []) or []))

    #     # prepare insertion vocabulary
    #     vocab = [w for w in getattr(self, "insertion_vocab", []) if w.isalpha() and w.lower() not in exclude_words]
    #     if not vocab or intensity == 0.0:
    #         return text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

    #     # --- 1) Protect excluded phrases with placeholders (case-insensitive) ---
    #     phrases_sorted = sorted(exclude_phrases, key=len, reverse=True)
    #     protected_map = {}
    #     protected_text = text

    #     for i, phrase in enumerate(phrases_sorted):
    #         if not phrase.strip():
    #             continue
    #         placeholder = f"__PHRASE_{i}__"

    #         def _sub(m):
    #             protected_map[placeholder] = m.group(0)  # preserve original casing
    #             return placeholder

    #         pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    #         protected_text = pattern.sub(_sub, protected_text)

    #     # --- 2) Tokenize ---
    #     tokens = tokenize(protected_text)

    #     # helpers
    #     def is_placeholder(tok: str) -> bool:
    #         return tok in protected_map

    #     def is_punct(tok: str) -> bool:
    #         return re.fullmatch(r"[^\w\s]", tok) is not None

    #     def is_word(tok: str) -> bool:
    #         # treat words incl. hyphen/apostrophe if your tokenizer keeps them whole
    #         return not is_placeholder(tok) and not is_punct(tok) and bool(re.search(r"[A-Za-z]", tok))

    #     # --- 3) Build eligible insertion positions ---
    #     # Positions are between tokens: 0..len(tokens)
    #     # Avoid inserting immediately adjacent to punctuation or touching placeholders (looks awkward)
    #     eligible_positions = []
    #     for pos in range(len(tokens) + 1):
    #         left = tokens[pos - 1] if pos - 1 >= 0 else None
    #         right = tokens[pos] if pos < len(tokens) else None

    #         # Disallow if adjacent to punctuation
    #         if (left and is_punct(left)) or (right and is_punct(right)):
    #             continue
    #         # Disallow if adjacent to a placeholder (keeps protected phrases tight)
    #         if (left and is_placeholder(left)) or (right and is_placeholder(right)):
    #             continue
    #         # Otherwise allow (including boundaries)
    #         eligible_positions.append(pos)

    #     if not eligible_positions:
    #         # nothing safe to insert into
    #         replaced_text = detokenize(tokens)
    #         for ph, original in protected_map.items():
    #             replaced_text = replaced_text.replace(ph, original)
    #         return replaced_text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

    #     # --- 4) Decide how many insertions (proportional to word count) ---
    #     n_words = sum(1 for t in tokens if is_word(t))
    #     k = int(round(n_words * intensity))
    #     k = max(0, min(k, len(eligible_positions)))
    #     if k == 0:
    #         replaced_text = detokenize(tokens)
    #         for ph, original in protected_map.items():
    #             replaced_text = replaced_text.replace(ph, original)
    #         return replaced_text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

    #     # sample positions without replacement
    #     chosen_positions = sorted(random.sample(eligible_positions, k), reverse=True)  # right→left to avoid index shifts

    #     # --- 5) Perform insertions ---
    #     inserted_words: List[str] = []
    #     insertion_positions: List[int] = []  # positions relative to original tokenization
    #     for pos in chosen_positions:
    #         w = random.choice(vocab)
    #         tokens.insert(pos, w)
    #         inserted_words.append(w)
    #         insertion_positions.append(pos)

    #     # --- 6) Detokenize and restore phrases ---
    #     replaced_text = detokenize(tokens)
    #     for ph, original in protected_map.items():
    #         replaced_text = replaced_text.replace(ph, original)

    #     return replaced_text, {
    #         "inserted_words": inserted_words[::-1],         # reverse to reflect left→right order if you prefer
    #         "insertion_positions": insertion_positions[::-1],
    #         "num_insertions": len(inserted_words)
    #     }



    def _apply_word_insertion(
        self,
        text: str,
        intensity: float,
        exclude_phrases: Optional[List[str]] = None,   # phrases to protect
        exclude_words: Optional[List[str]] = None,     # words never to insert
    ) -> Tuple[str, Dict]:
        """
        Insert words in grammatically sensible places using POS:
        - before adjectives (JJ*): use intensifiers (very, particularly, …)
        - before nouns (NN*): use specificity/importance (specific, key, …)
        - before verbs (VB*): use frequency/temporal (regularly, periodically, …)
        Honors exclude_phrases/words and avoids punctuation/placeholder adjacency.
        """

        # --- config / normalization ---
        intensity = max(0.0, min(1.0, float(intensity)))
        exclude_phrases = exclude_phrases if exclude_phrases is not None else getattr(self, "default_exclude_phrases", []) or []
        exclude_words = set(w.lower() for w in (exclude_words if exclude_words is not None else getattr(self, "default_exclude_words", []) or []))
        # print(f"Excluding phrases: {exclude_phrases}, words: {exclude_words}")
        # Split your insertion vocab into contextual sublists (fallback to the class list if present)
        base_vocab = list(getattr(self, "insertion_vocab", []))
        # If you want to hardwire from your earlier list, you could set insertion_vocab on self.

        # Buckets tuned for policy/procedure tone
        intensifiers = [w for w in base_vocab if w in {
            "very","quite","really","actually","particularly","especially","significantly","notably"
        }]
        specificity  = [w for w in base_vocab if w in {
            "specific","certain","various","relevant","appropriate","applicable","designated","related"
        }]
        importance   = [w for w in base_vocab if w in {
            "important","main","key","primary","critical","essential","necessary","mandatory"
        }]
        frequency    = [w for w in base_vocab if w in {
            "generally","typically","normally","commonly","regularly","usually","periodically","routinely"
        }]

        # Context → vocab mapping
        POS_INSERT_VOCAB = {
            "JJ": intensifiers or base_vocab,                    # before adjectives
            "NN": (specificity + importance) or base_vocab,      # before nouns
            "VB": frequency or base_vocab,                       # before verbs
        }

        # Filter by exclude_words and non-alpha
        def filter_vocab(xs: List[str]) -> List[str]:
            return [w for w in xs if w.isalpha() and w.lower() not in exclude_words]

        for k in POS_INSERT_VOCAB:
            POS_INSERT_VOCAB[k] = filter_vocab(POS_INSERT_VOCAB[k])
        base_vocab = filter_vocab(base_vocab)

        # If nothing usable, bail early
        if (not base_vocab) or all(not v for v in POS_INSERT_VOCAB.values()):
            return text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

        # --- 1) Protect excluded phrases (case-insensitive), store matched form for case preservation ---
        phrases_sorted = sorted(exclude_phrases, key=len, reverse=True)
        protected_map: Dict[str, str] = {}
        protected_text = text

        for i, phrase in enumerate(phrases_sorted):
            if not phrase.strip():
                continue
            placeholder = f"__PHRASE_{i}__"

            def _sub(m):
                protected_map[placeholder] = m.group(0)
                return placeholder

            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            protected_text = pattern.sub(_sub, protected_text)

        # --- 2) Tokenize ---
        tokens = tokenize(protected_text)

        # helpers
        def is_placeholder(tok: str) -> bool:
            return tok in protected_map

        def is_punct(tok: str) -> bool:
            return re.fullmatch(r"[^\w\s]", tok) is not None

        def is_word(tok: str) -> bool:
            return (not is_placeholder(tok)) and (not is_punct(tok)) and bool(re.search(r"[A-Za-z]", tok))

        # --- 3) Build POS tags for word tokens (keep index mapping) ---
        word_idxs: List[int] = []
        word_strs: List[str] = []
        for i, t in enumerate(tokens):
            if is_word(t):
                word_idxs.append(i)
                word_strs.append(t)

        if not word_strs or intensity == 0.0:
            # Nothing to do
            replaced_text = detokenize(tokens)
            for ph, original in protected_map.items():
                replaced_text = replaced_text.replace(ph, original)
            return replaced_text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

        tagged = nltk.pos_tag(word_strs)  # List[ (token, POS) ] with Penn Treebank tags

        # --- 4) Build eligible insertion positions with POS context ---
        # We insert BEFORE a token if its POS is in target sets, and the boundary is "clean"
        def allowed_boundary(pos_between: int) -> bool:
            # Disallow if adjacent to punctuation or placeholders
            left = tokens[pos_between - 1] if pos_between - 1 >= 0 else None
            right = tokens[pos_between] if pos_between < len(tokens) else None
            if (left and (is_punct(left) or is_placeholder(left))) or (right and (is_punct(right) or is_placeholder(right))):
                return False
            return True

        candidate_positions: List[tuple[int, str]] = []  # (insert_pos, POS_tag_of_right_token)
        for (tok_idx_in_tokens, (_tok, pos_tag)) in zip(word_idxs, tagged):
            # Collapse to families JJ/NN/VB for mapping
            family = "JJ" if pos_tag.startswith("JJ") else "NN" if pos_tag.startswith("NN") else "VB" if pos_tag.startswith("VB") else None
            if family and POS_INSERT_VOCAB.get(family):
                insert_pos = tok_idx_in_tokens  # insert BEFORE this token
                if allowed_boundary(insert_pos):
                    candidate_positions.append((insert_pos, family))

        # Fallback: if no POS-friendly spots, allow generic clean boundaries anywhere
        if not candidate_positions:
            generic_positions = []
            for pos in range(len(tokens) + 1):
                if allowed_boundary(pos):
                    generic_positions.append((pos, None))
            candidate_positions = generic_positions

        if not candidate_positions:
            replaced_text = detokenize(tokens)
            for ph, original in protected_map.items():
                replaced_text = replaced_text.replace(ph, original)
            return replaced_text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

        # --- 5) Decide how many insertions (proportional, deterministic by count) ---
        n_words = len(word_strs)
        k = int(round(n_words * intensity))
        k = max(0, min(k, len(candidate_positions)))
        if k == 0:
            replaced_text = detokenize(tokens)
            for ph, original in protected_map.items():
                replaced_text = replaced_text.replace(ph, original)
            return replaced_text, {"inserted_words": [], "insertion_positions": [], "num_insertions": 0}

        # Sample positions without replacement; insert from right→left to avoid index shifts
        chosen = sorted(random.sample(candidate_positions, k), key=lambda x: x[0], reverse=True)

        inserted_words: List[str] = []
        insertion_positions: List[int] = []

        for insert_pos, family in chosen:
            # Pick a word suitable for the POS family, with fallback to base vocab
            pool = POS_INSERT_VOCAB.get(family) or base_vocab
            if not pool:
                continue
            w = random.choice(pool)
            tokens.insert(insert_pos, w)
            inserted_words.append(w)
            insertion_positions.append(insert_pos)

        # --- 6) Detokenize and restore phrases ---
        replaced_text = detokenize(tokens)
        for ph, original in protected_map.items():
            replaced_text = replaced_text.replace(ph, original)

        # (Optional) reverse lists if you prefer left→right reporting
        inserted_words.reverse()
        insertion_positions.reverse()

        return replaced_text, {
            "inserted_words": inserted_words,
            "insertion_positions": insertion_positions,
            "num_insertions": len(inserted_words),
        }



    def _apply_word_reordering(
        self,
        text: str,
        intensity: float,
        exclude_phrases: Optional[List[str]] = None,
        exclude_words: Optional[List[str]] = None,
        preserve_edges: bool = True,
    ) -> Tuple[str, Dict]:
        """Reorder words by swapping adjacent candidate pairs only."""

        intensity = max(0.0, min(1.0, float(intensity)))
        exclude_phrases = exclude_phrases if exclude_phrases is not None else getattr(self, "default_exclude_phrases", []) or []
        exclude_words = set(w.lower() for w in (exclude_words if exclude_words is not None else getattr(self, "default_exclude_words", []) or []))
        stop_words = set(w.lower() for w in getattr(self, "stop_words", set()))

        # 1) Protect excluded phrases (case-insensitive)
        phrases_sorted = sorted(exclude_phrases, key=len, reverse=True)
        protected_map, protected_text = {}, text
        for i, phrase in enumerate(phrases_sorted):
            if not phrase.strip():
                continue
            placeholder = f"__PHRASE_{i}__"
            def _sub(m):
                protected_map[placeholder] = m.group(0)
                return placeholder
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            protected_text = pattern.sub(_sub, protected_text)

        # 2) Tokenize (assumes tokenize/detokenize helpers as before)
        tokens = tokenize(protected_text)

        def is_placeholder(tok: str) -> bool:
            return tok in protected_map
        def is_punct(tok: str) -> bool:
            return re.fullmatch(r"[^\w\s]", tok) is not None
        def is_word(tok: str) -> bool:
            return (not is_placeholder(tok)) and (not is_punct(tok)) and bool(re.search(r"[A-Za-z]", tok))

        # 3) Candidate word indices (exclude stop/excluded)
        cand_idxs = [i for i, tok in enumerate(tokens)
                    if is_word(tok) and tok.lower() not in stop_words and tok.lower() not in exclude_words]

        # Optionally protect first/last token positions
        if preserve_edges:
            if cand_idxs and cand_idxs[0] == 0:
                cand_idxs = cand_idxs[1:]
            # Remove or comment out the next block so the last word is NOT preserved:
            # if cand_idxs and cand_idxs[-1] == len(tokens) - 1:
            #     cand_idxs = cand_idxs[:-1]

        # Nothing to do?
        if len(cand_idxs) < 2 or intensity == 0.0:
            out = detokenize(tokens)
            for ph, orig in protected_map.items():
                out = out.replace(ph, orig)
            return out, {"affected_words": [], "num_swaps": 0, "swapped_pairs": []}

        # 4) Form **non-overlapping adjacent pairs** in original order
        #    Example: indices [3,5,7,10,12] -> pairs: (3,5), (7,10)  (12 is leftover)
        max_pairs = len(cand_idxs) // 2
        adj_pairs = [(cand_idxs[2*j], cand_idxs[2*j+1]) for j in range(max_pairs)]

        # 5) Decide how many pairs to swap (proportional, deterministic by count)
        k_pairs = int(round(max_pairs * intensity))
        k_pairs = max(0, min(k_pairs, max_pairs))
        if k_pairs == 0:
            out = detokenize(tokens)
            for ph, orig in protected_map.items():
                out = out.replace(ph, orig)
            return out, {"affected_words": [], "num_swaps": 0, "swapped_pairs": []}

        # Randomize which adjacent pairs get swapped, but pairs remain adjacent by construction
        random.shuffle(adj_pairs)
        chosen_pairs = adj_pairs[:k_pairs]

        # 6) Perform swaps; record ORIGINAL tokens and indices
        swapped_pairs = []
        for i1, i2 in chosen_pairs:
            if i1 == i2:
                continue
            w1, w2 = tokens[i1], tokens[i2]
            tokens[i1], tokens[i2] = w2, w1
            swapped_pairs.append((i1, i2, w1, w2))

        # 7) Detokenize and restore phrases
        out = detokenize(tokens)
        for ph, orig in protected_map.items():
            out = out.replace(ph, orig)

        affected = [f"{w1}↔{w2}" for (_, _, w1, w2) in swapped_pairs]
        return out, {
            "affected_words": affected,
            "num_swaps": len(swapped_pairs),
            "swapped_pairs": swapped_pairs,  # (idx1, idx2, orig1, orig2)
        }


    
    # def _apply_word_reordering(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Reorder words in the query"""
    #     words = text.split()
    #     if len(words) <= 1:
    #         return text, {'affected_words': [], 'num_swaps': 0}
        
    #     num_swaps = max(1, int(len(words) * intensity / 2))
    #     swapped_pairs = []
        
    #     for _ in range(num_swaps):
    #         if len(words) < 2:
    #             break
            
    #         # Select two different positions
    #         idx1 = random.randint(0, len(words) - 1)
    #         idx2 = random.randint(0, len(words) - 1)
    #         while idx2 == idx1:
    #             idx2 = random.randint(0, len(words) - 1)
            
    #         # Swap
    #         words[idx1], words[idx2] = words[idx2], words[idx1]
    #         swapped_pairs.append((idx1, idx2, words[idx1], words[idx2]))
        
    #     return ' '.join(words), {
    #         'swapped_pairs': swapped_pairs,
    #         'num_swaps': len(swapped_pairs)
    #     }
    
    # def _apply_word_duplication(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Duplicate random words"""
    #     words = text.split()
    #     num_duplications = max(1, int(len(words) * intensity))
        
    #     duplicated_words = []
    #     duplication_positions = []
        
    #     # Work backwards to avoid index shifting issues
    #     positions = random.sample(range(len(words)), 
    #                             min(num_duplications, len(words)))
        
    #     for pos in sorted(positions, reverse=True):
    #         words.insert(pos + 1, words[pos])
    #         duplicated_words.append(words[pos])
    #         duplication_positions.append(pos)
        
    #     return ' '.join(words), {
    #         'duplicated_words': duplicated_words,
    #         'duplication_positions': duplication_positions,
    #         'num_duplications': len(duplicated_words)
    #     }
    
    # def _apply_stop_word_removal(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Remove stop words"""
    #     words = text.split()
    #     stop_words_in_text = [(i, w) for i, w in enumerate(words) 
    #                          if w.lower() in self.stop_words]
        
    #     if not stop_words_in_text:
    #         return text, {'removed_words': [], 'num_removed': 0}
        
    #     num_to_remove = max(1, int(len(stop_words_in_text) * intensity))
    #     to_remove = random.sample(stop_words_in_text, 
    #                             min(num_to_remove, len(stop_words_in_text)))
        
    #     indices_to_remove = {idx for idx, _ in to_remove}
    #     removed_words = [word for _, word in to_remove]
        
    #     remaining_words = [w for i, w in enumerate(words) 
    #                       if i not in indices_to_remove]
        
    #     return ' '.join(remaining_words), {
    #         'removed_words': removed_words,
    #         'num_removed': len(removed_words)
    #     }
    
    # def _apply_stemming_variation(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Apply stemming variations"""
    #     words = text.split()
    #     num_variations = max(1, int(len(words) * intensity))
        
    #     # Simple stemming variations
    #     variations = {
    #         'ing': ['', 'e', 'ed'],
    #         'ed': ['', 'e', 'ing'],
    #         's': ['', 'es'],
    #         'es': ['', 's'],
    #         'ies': ['y', 'ie'],
    #         'tion': ['te', 't', 'ting'],
    #         'ment': ['', 'ing']
    #     }
        
    #     affected_words = []
    #     original_forms = {}
        
    #     candidates = [(i, w) for i, w in enumerate(words) if len(w) > 3]
    #     if candidates:
    #         selected = random.sample(candidates, 
    #                                min(num_variations, len(candidates)))
            
    #         for idx, word in selected:
    #             for suffix, replacements in variations.items():
    #                 if word.lower().endswith(suffix):
    #                     new_ending = random.choice(replacements)
    #                     new_word = word[:-len(suffix)] + new_ending
                        
    #                     # Preserve capitalization
    #                     if word[0].isupper():
    #                         new_word = new_word.capitalize()
                        
    #                     affected_words.append(word)
    #                     original_forms[new_word] = word
    #                     words[idx] = new_word
    #                     break
        
    #     return ' '.join(words), {
    #         'affected_words': affected_words,
    #         'variations': original_forms,
    #         'num_variations': len(affected_words)
    #     }
    
    # def _apply_hyponym_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Replace words with more specific terms (hyponyms)"""
    #     words = text.split()
    #     num_replacements = max(1, int(len(words) * intensity))
        
    #     affected_words = []
    #     replacements = {}
        
    #     content_words = [(i, w) for i, w in enumerate(words) 
    #                     if w.lower() not in self.stop_words and len(w) > 2]
        
    #     if content_words:
    #         selected = random.sample(content_words, 
    #                                min(num_replacements, len(content_words)))
            
    #         for idx, word in selected:
    #             synsets = wordnet.synsets(word.lower(), pos=self._get_wordnet_pos(word))
    #             hyponyms = []
                
    #             for synset in synsets[:2]:  # Check first 2 synsets
    #                 for hyponym in synset.hyponyms():
    #                     hyponym_name = hyponym.lemmas()[0].name().replace('_', ' ')
    #                     if hyponym_name.lower() != word.lower():
    #                         hyponyms.append(hyponym_name)
                
    #             if hyponyms:
    #                 replacement = random.choice(hyponyms[:3])
    #                 if word[0].isupper():
    #                     replacement = replacement.capitalize()
                    
    #                 affected_words.append(word)
    #                 replacements[word] = replacement
    #                 words[idx] = replacement
        
    #     return ' '.join(words), {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }
    
    # def _apply_hypernym_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Replace words with more general terms (hypernyms)"""
    #     words = text.split()
    #     num_replacements = max(1, int(len(words) * intensity))
        
    #     affected_words = []
    #     replacements = {}
        
    #     content_words = [(i, w) for i, w in enumerate(words) 
    #                     if w.lower() not in self.stop_words and len(w) > 2]
        
    #     if content_words:
    #         selected = random.sample(content_words, 
    #                                min(num_replacements, len(content_words)))
            
    #         for idx, word in selected:
    #             synsets = wordnet.synsets(word.lower(), pos=self._get_wordnet_pos(word))
    #             hypernyms = []
                
    #             for synset in synsets[:2]:  # Check first 2 synsets
    #                 for hypernym in synset.hypernyms():
    #                     hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
    #                     if hypernym_name.lower() != word.lower():
    #                         hypernyms.append(hypernym_name)
                
    #             if hypernyms:
    #                 replacement = random.choice(hypernyms[:3])
    #                 if word[0].isupper():
    #                     replacement = replacement.capitalize()
                    
    #                 affected_words.append(word)
    #                 replacements[word] = replacement
    #                 words[idx] = replacement
        
    #     return ' '.join(words), {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }
    
    # def _apply_related_word_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Replace words with semantically related words"""
    #     words = text.split()
    #     num_replacements = max(1, int(len(words) * intensity))
        
    #     # Domain-specific related words
    #     domain_relations = {
    #         'search': ['query', 'find', 'lookup', 'seek', 'retrieve'],
    #         'algorithm': ['method', 'procedure', 'technique', 'approach'],
    #         'model': ['system', 'framework', 'architecture', 'design'],
    #         'data': ['information', 'content', 'dataset', 'records'],
    #         'user': ['person', 'individual', 'customer', 'client'],
    #         'result': ['output', 'outcome', 'finding', 'response'],
    #         'document': ['file', 'text', 'page', 'article', 'content'],
    #         'query': ['question', 'search', 'request', 'inquiry'],
    #         'ranking': ['ordering', 'sorting', 'scoring', 'positioning'],
    #         'relevance': ['importance', 'significance', 'pertinence', 'applicability']
    #     }
        
    #     affected_words = []
    #     replacements = {}
        
    #     content_words = [(i, w) for i, w in enumerate(words) 
    #                     if w.lower() not in self.stop_words]
        
    #     if content_words:
    #         selected = random.sample(content_words, 
    #                                min(num_replacements, len(content_words)))
            
    #         for idx, word in selected:
    #             word_lower = word.lower()
                
    #             # Check domain relations
    #             if word_lower in domain_relations:
    #                 related = domain_relations[word_lower]
    #             else:
    #                 # Use semantic replacements as fallback
    #                 related = self.semantic_replacements.get(word_lower, [])
                
    #             if related:
    #                 replacement = random.choice(related)
    #                 if word[0].isupper():
    #                     replacement = replacement.capitalize()
                    
    #                 affected_words.append(word)
    #                 replacements[word] = replacement
    #                 words[idx] = replacement
        
    #     return ' '.join(words), {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }
    
    # def _apply_random_word_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Replace words with random vocabulary words"""
    #     words = text.split()
    #     num_replacements = max(1, int(len(words) * intensity))
        
    #     # Random vocabulary for replacement
    #     random_vocab = [
    #         'system', 'process', 'method', 'approach', 'technique',
    #         'framework', 'model', 'algorithm', 'data', 'information',
    #         'analysis', 'structure', 'function', 'element', 'component',
    #         'feature', 'attribute', 'property', 'characteristic', 'aspect'
    #     ]
        
    #     affected_words = []
    #     replacements = {}
        
    #     positions = random.sample(range(len(words)), 
    #                             min(num_replacements, len(words)))
        
    #     for pos in positions:
    #         original = words[pos]
    #         replacement = random.choice(random_vocab)
            
    #         # Preserve capitalization
    #         if original[0].isupper():
    #             replacement = replacement.capitalize()
            
    #         affected_words.append(original)
    #         replacements[original] = replacement
    #         words[pos] = replacement
        
    #     return ' '.join(words), {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }
    


    # def _apply_random_word_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
        
    #     """Replace words with any random English word"""
    #     words_in_text = tokenize(text)
    #     num_replacements = max(1, int(len(words_in_text) * intensity))

    #     # Use NLTK's words corpus as the pool
    #     random_word_pool = [w for w in nltk_words.words() if w.isalpha() and 2 < len(w) < 30]
    #     print(random_word_pool)
    #     affected_words = []
    #     replacements = {}

    #     positions = random.sample(range(len(words_in_text)), min(num_replacements, len(words_in_text)))

    #     for pos in positions:
    #         original = words_in_text[pos]
    #         replacement = random.choice(random_word_pool)
    #         # Preserve capitalization
    #         if original[0].isupper():
    #             replacement = replacement.capitalize()
    #         affected_words.append(original)
    #         replacements[original] = replacement
    #         words_in_text[pos] = replacement

    #     return detokenize(words_in_text), {
    #         'affected_words': affected_words,
    #         'replacements': replacements,
    #         'num_replacements': len(affected_words)
    #     }

    def _apply_random_word_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Replace words with any random English word"""

        if not text:
            return text, {
                'affected_words': [],
                'replacements': {},
                'num_replacements': 0
            }

        words_in_text = tokenize(text)
        if not words_in_text:
            return text, {
                'affected_words': [],
                'replacements': {},
                'num_replacements': 0
            }

        # Clip intensity to [0, 1]
        intensity = max(0.0, min(1.0, float(intensity)))
        num_replacements = max(1, int(len(words_in_text) * intensity))

        # Cache the random word pool for performance
        if not hasattr(self, "_random_word_pool"):
            from nltk.corpus import words as nltk_words
            self._random_word_pool = [
                w for w in nltk_words.words()
                if w.isalpha() and 3 <= len(w) <= 20
            ]

        affected_words = []
        replacements = {}

        # Positions to replace (only alphabetic tokens)
        eligible_positions = [i for i, tok in enumerate(words_in_text) if tok.isalpha()]
        if not eligible_positions:
            return text, {
                'affected_words': [],
                'replacements': {},
                'num_replacements': 0
            }

        positions = random.sample(eligible_positions, min(num_replacements, len(eligible_positions)))

        def match_casing(src: str, tgt: str) -> str:
            if src.isupper():
                return tgt.upper()
            elif src.istitle():
                return tgt.title()
            elif src[0].isupper():
                return tgt.capitalize()
            else:
                return tgt.lower()

        for pos in positions:
            original = words_in_text[pos]
            replacement = random.choice(self._random_word_pool)
            while replacement.lower() == original.lower():
                replacement = random.choice(self._random_word_pool)
            replacement = match_casing(original, replacement)

            affected_words.append(original)
            replacements[original] = replacement
            words_in_text[pos] = replacement
        # print("word in text:", words_in_text)
        return detokenize(words_in_text), {
            'affected_words': affected_words,
            'replacements': replacements,
            'num_replacements': len(affected_words)
        }



    # def _apply_random_word_replacement(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """
    #     Replace a fraction of word tokens with random English words.
    #     - intensity: fraction of word tokens to replace (clipped to [0, 1]).
    #     - Only alphabetic tokens are eligible (skip numbers, punctuation, URLs, etc.).
    #     Returns: (detokenized_text, meta)
    #     meta = {
    #         'num_candidates': <eligible word tokens>,
    #         'num_replacements': <actually replaced>,
    #         'replacements': [{'pos': i, 'original': o, 'replacement': r}, ...]
    #     }
    #     """
    #     # --- fast exits & bounds ---
    #     if not text:
    #         return text, {'num_candidates': 0, 'num_replacements': 0, 'replacements': []}
    #     intensity = max(0.0, min(1.0, float(intensity)))

    #     tokens = tokenize(text)  # must be whitespace- & punctuation-preserving
    #     n = len(tokens)
    #     if n == 0 or intensity == 0.0:
    #         return text, {'num_candidates': 0, 'num_replacements': 0, 'replacements': []}

    #     # --- build/cache pool once in __init__ ideally ---
    #     # Expect: self.random_word_pool precomputed; fallback shown if not.
    #     pool = getattr(self, "random_word_pool", None)
    #     if pool is None:
    #         from nltk.corpus import words as nltk_words
    #         # Moderate frequency proxy: length filter + alpha-only
    #         pool = [w for w in nltk_words.words() if w.isalpha() and 3 <= len(w) <= 20]
    #         self.random_word_pool = pool

    #     # Helper: identify eligible tokens (alphabetic only)
    #     def is_word(tok: str) -> bool:
    #         # tighten with regex if you need to skip emails/urls: r"^[A-Za-z]+$"
    #         return tok.isalpha()

    #     word_positions = [i for i, t in enumerate(tokens) if is_word(t)]
    #     num_candidates = len(word_positions)
    #     if num_candidates == 0:
    #         return text, {'num_candidates': 0, 'num_replacements': 0, 'replacements': []}

    #     k = max(1, int(num_candidates * intensity))
    #     positions = random.sample(word_positions, k=min(k, num_candidates))

    #     # Casing utility
    #     def match_casing(src: str, tgt: str) -> str:
    #         if src.isupper():
    #             return tgt.upper()
    #         if src.istitle():
    #             return tgt.title()
    #         if src[0].isupper():
    #             return tgt.capitalize()
    #         return tgt.lower()

    #     # Pre-sample replacements (without replacement for variety)
    #     replacements_sample = random.sample(pool, k=len(positions))

    #     meta = {'num_candidates': num_candidates, 'num_replacements': 0, 'replacements': []}
    #     for pos, candidate in zip(positions, replacements_sample):
    #         original = tokens[pos]
    #         # avoid identity replacement ignoring case
    #         if candidate.lower() == original.lower():
    #             # retry once; if same, skip
    #             candidate = random.choice(pool)
    #             if candidate.lower() == original.lower():
    #                 continue
    #         tokens[pos] = match_casing(original, candidate)
    #         meta['replacements'].append({'pos': pos, 'original': original, 'replacement': tokens[pos]})
    #         meta['num_replacements'] += 1

    #     return detokenize(tokens), meta







    # def _apply_phrase_paraphrasing(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Paraphrase common phrases"""
    #     # Common phrase paraphrases
    #     phrase_paraphrases = {
    #         'machine learning': ['ML', 'automated learning', 'computational learning'],
    #         'natural language': ['human language', 'linguistic', 'NL'],
    #         'information retrieval': ['IR', 'data retrieval', 'info search'],
    #         'neural network': ['NN', 'neural net', 'artificial neural network'],
    #         'deep learning': ['DL', 'deep neural learning', 'hierarchical learning'],
    #         'search engine': ['search system', 'retrieval engine', 'search tool'],
    #         'data mining': ['knowledge discovery', 'data extraction', 'pattern mining'],
    #         'big data': ['large-scale data', 'massive datasets', 'voluminous data']
    #     }
        
    #     affected_phrases = []
    #     replacements = {}
        
    #     # Determine how many phrases to replace based on intensity
    #     for phrase, alternatives in phrase_paraphrases.items():
    #         if phrase in text.lower() and random.random() < intensity:
    #             replacement = random.choice(alternatives)
                
    #             # Handle case preservation
    #             if phrase in text:
    #                 text = text.replace(phrase, replacement)
    #             elif phrase.title() in text:
    #                 text = text.replace(phrase.title(), replacement.title())
    #             elif phrase.upper() in text:
    #                 text = text.replace(phrase.upper(), replacement.upper())
                
    #             affected_phrases.append(phrase)
    #             replacements[phrase] = replacement
        
    #     return text, {
    #         'affected_phrases': affected_phrases,
    #         'replacements': replacements,
    #         'num_paraphrases': len(affected_phrases)
    #     }


    def _apply_word_splitting(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Randomly split words at arbitrary positions to mimic typing errors."""
        intensity = max(0.0, min(1.0, float(intensity)))
        words = tokenize(text)

        if not words or intensity == 0.0:
            return text, {"affected_words": [], "splits": {}, "num_splits": 0}

        print(words)
        n = len(words)
        k = int(round(n * intensity))
        k = max(0, min(k, n))

        chosen_indices = random.sample(range(n), k)
        affected_words: List[str] = []
        splits: Dict[str, str] = {}

        for idx in sorted(chosen_indices, reverse=True):  # reverse to avoid messing indices
            word = words[idx]
            if len(word) < 2:
                continue
            split_pos = random.randint(1, len(word) - 1)  # avoid splitting at 0 or end
            split_version = word[:split_pos] + " " + word[split_pos:]
            words[idx:idx+1] = split_version.split()  # replace with two tokens
            affected_words.append(word)
            splits[word] = split_version

        return detokenize(words), {
            "affected_words": affected_words,
            "splits": splits,
            "num_splits": len(affected_words)
        }












    # def _apply_word_splitting(self, text: str, intensity: float) -> Tuple[str, Dict]:
    #     """Split compound words"""
    #     words = text.split()
        
    #     # Common compound words that can be split
    #     splittable_compounds = {
    #         'database': 'data base',
    #         'dataset': 'data set',
    #         'framework': 'frame work',
    #         'workflow': 'work flow',
    #         'endpoint': 'end point',
    #         'backend': 'back end',
    #         'frontend': 'front end',
    #         'metadata': 'meta data',
    #         'timestamp': 'time stamp',
    #         'username': 'user name',
    #         'filename': 'file name',
    #         'keyword': 'key word',
    #         'substring': 'sub string',
    #         'subquery': 'sub query'
    #     }
        
    #     affected_words = []
    #     splits = {}
    #     new_words = []
        
    #     for word in words:
    #         word_lower = word.lower()
    #         if word_lower in splittable_compounds and random.random() < intensity:
    #             split_version = splittable_compounds[word_lower]
                
    #             # Preserve capitalization
    #             if word[0].isupper():
    #                 split_parts = split_version.split()
    #                 split_parts[0] = split_parts[0].capitalize()
    #                 split_version = ' '.join(split_parts)
                
    #             new_words.extend(split_version.split())
    #             affected_words.append(word)
    #             splits[word] = split_version
    #         else:
    #             new_words.append(word)
        
    #     return ' '.join(new_words), {
    #         'affected_words': affected_words,
    #         'splits': splits,
    #         'num_splits': len(affected_words)
    #     }
    
    def _apply_word_merging(self, text: str, intensity: float) -> Tuple[str, Dict]:
        """Merge adjacent words anywhere in the text with probability = intensity per boundary."""
        words = tokenize(text)
        if len(words) < 2:
            return text, {'merged_pairs': [], 'merges': {}, 'num_merges': 0}

        # Clamp intensity to [0, 1]
        intensity = max(0.0, min(1.0, float(intensity)))

        merged_pairs = []
        merges = {}
        new_words = []
        i = 0

        while i < len(words):
            if i < len(words) - 1 and random.random() < intensity:
                # Merge this pair (no overlap with next, since we skip i+1)
                merged = words[i] + words[i + 1].lower()  # keep behavior: second token lowercased
                new_words.append(merged)
                merged_pairs.append((words[i], words[i + 1]))
                merges[f"{words[i]} {words[i + 1]}"] = merged
                i += 2
            else:
                new_words.append(words[i])
                i += 1

        return detokenize(new_words), {
            'merged_pairs': merged_pairs,
            'merges': merges,
            'num_merges': len(merged_pairs)
        }


class WordLevelIRModelTester:
    """Main testing framework for word-level perturbations"""
    
    def __init__(self, model_interface: Callable):
        """
        Initialize tester with model interface
        
        Args:
            model_interface: Function that takes query and returns ranked results
        """
        self.model = model_interface
        self.perturbator = WordPerturbator()
        self.test_results = defaultdict(list)
    
    def generate_test_cases(self, queries: List[str], 
                           perturbation_types: List[WordPerturbationType] = None,
                           intensity_levels: List[float] = None) -> List[WordTestCase]:
        """Generate word-level test cases"""
        if perturbation_types is None:
            perturbation_types = list(WordPerturbationType)
        
        if intensity_levels is None:
            intensity_levels = [0.1, 0.2, 0.3, 0.5, 0.7]
        
        test_cases = []
        
        for query in queries:
            for p_type in perturbation_types:
                for intensity in intensity_levels:
                    perturbed, details = self.perturbator.apply_perturbation(
                        query, p_type, intensity
                    )
                    
                    test_cases.append(WordTestCase(
                        original_query=query,
                        perturbed_query=perturbed,
                        perturbation_type=p_type,
                        intensity=intensity,
                        affected_words=details.get('affected_words', []),
                        perturbation_details=details
                    ))
        
        return test_cases
    
    def run_sensitivity_test(self, test_cases: List[WordTestCase]) -> Dict:
        """Run word-level sensitivity tests"""
        results = {
            'summary': {},
            'details': [],
            'by_perturbation': defaultdict(list),
            'by_intensity': defaultdict(list),
            'semantic_analysis': {}
        }
        
        for test_case in test_cases:
            # Get results for original and perturbed queries
            original_results = self.model(test_case.original_query)
            perturbed_results = self.model(test_case.perturbed_query)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                original_results, 
                perturbed_results,
                test_case
            )
            
            # Store results
            result = {
                'test_case': test_case,
                'metrics': metrics,
                'degradation': metrics['overall_degradation']
            }
            
            results['details'].append(result)
            results['by_perturbation'][test_case.perturbation_type].append(result)
            results['by_intensity'][test_case.intensity].append(result)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary(results)
        results['semantic_analysis'] = self._analyze_semantic_impact(results)
        
        return results
    
    def _calculate_metrics(self, original: List[Tuple], perturbed: List[Tuple], 
                          test_case: WordTestCase) -> Dict:
        """Calculate word-level specific metrics"""
        metrics = {}
        
        # Basic overlap and ranking metrics
        orig_docs = [doc_id for doc_id, _ in original]
        pert_docs = [doc_id for doc_id, _ in perturbed]
        
        common_docs = set(orig_docs) & set(pert_docs)
        metrics['overlap_ratio'] = len(common_docs) / len(orig_docs) if orig_docs else 0
        
        # Precision at different cutoffs
        for k in [1, 3, 5, 10]:
            if len(orig_docs) >= k and len(pert_docs) >= k:
                orig_top_k = set(orig_docs[:k])
                pert_top_k = set(pert_docs[:k])
                metrics[f'precision_at_{k}'] = len(orig_top_k & pert_top_k) / k
            else:
                metrics[f'precision_at_{k}'] = 0
        
        # Rank correlation metrics
        if common_docs:
            orig_ranks = {doc: i for i, doc in enumerate(orig_docs)}
            pert_ranks = {doc: i for i, doc in enumerate(pert_docs)}
            
            rank_diffs = []
            for doc in common_docs:
                if doc in orig_ranks and doc in pert_ranks:
                    rank_diffs.append(abs(orig_ranks[doc] - pert_ranks[doc]))
            
            metrics['avg_rank_diff'] = np.mean(rank_diffs) if rank_diffs else 0
            metrics['max_rank_diff'] = max(rank_diffs) if rank_diffs else 0
            
            # Normalized Discounted Cumulative Gain (NDCG) approximation
            dcg_orig = sum(1.0 / np.log2(i + 2) for i in range(len(orig_docs)))
            dcg_pert = 0
            for i, doc in enumerate(pert_docs):
                if doc in orig_docs:
                    orig_pos = orig_docs.index(doc)
                    dcg_pert += 1.0 / np.log2(i + 2) * (1.0 / np.log2(orig_pos + 2))
            
            metrics['ndcg_approx'] = dcg_pert / dcg_orig if dcg_orig > 0 else 0
        else:
            metrics['avg_rank_diff'] = len(orig_docs)
            metrics['max_rank_diff'] = len(orig_docs)
            metrics['ndcg_approx'] = 0
        
        # Word-level specific metrics
        metrics['num_words_affected'] = len(test_case.affected_words)
        metrics['word_change_ratio'] = (len(test_case.affected_words) / 
                                       len(test_case.original_query.split()))
        
        # Query length change
        orig_length = len(test_case.original_query.split())
        pert_length = len(test_case.perturbed_query.split())
        metrics['length_change_ratio'] = abs(orig_length - pert_length) / orig_length
        
        # Overall degradation score (weighted combination)
        metrics['overall_degradation'] = (
            0.3 * (1 - metrics['overlap_ratio']) +
            0.2 * (1 - metrics['precision_at_5']) +
            0.2 * (1 - metrics['ndcg_approx']) +
            0.2 * (metrics['avg_rank_diff'] / len(orig_docs) if orig_docs else 1) +
            0.1 * metrics['length_change_ratio']
        )
        
        return metrics
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics for word-level tests"""
        summary = {}
        
        # Overall statistics
        all_degradations = [r['degradation'] for r in results['details']]
        summary['avg_degradation'] = np.mean(all_degradations)
        summary['std_degradation'] = np.std(all_degradations)
        summary['max_degradation'] = max(all_degradations)
        summary['min_degradation'] = min(all_degradations)
        
        # Precision metrics summary
        for k in [1, 3, 5, 10]:
            precisions = [r['metrics'][f'precision_at_{k}'] 
                         for r in results['details']]
            summary[f'avg_precision_at_{k}'] = np.mean(precisions)
        
        # By perturbation type
        summary['by_perturbation'] = {}
        for p_type, results_list in results['by_perturbation'].items():
            degradations = [r['degradation'] for r in results_list]
            summary['by_perturbation'][p_type.value] = {
                'avg': np.mean(degradations),
                'std': np.std(degradations),
                'count': len(degradations),
                'avg_words_affected': np.mean([r['metrics']['num_words_affected'] 
                                              for r in results_list])
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
    
    def _analyze_semantic_impact(self, results: Dict) -> Dict:
        """Analyze the semantic impact of different perturbation types"""
        semantic_analysis = {
            'high_impact_perturbations': [],
            'low_impact_perturbations': [],
            'semantic_vs_syntactic': {},
            'recommendations': []
        }
        
        # Categorize perturbations by semantic impact
        semantic_perturbations = {
            WordPerturbationType.SYNONYM_REPLACEMENT,
            WordPerturbationType.HYPONYM_REPLACEMENT,
            WordPerturbationType.HYPERNYM_REPLACEMENT,
            WordPerturbationType.RELATED_WORD_REPLACEMENT,
            WordPerturbationType.PHRASE_PARAPHRASING
        }
        
        syntactic_perturbations = {
            WordPerturbationType.WORD_DELETION,
            WordPerturbationType.WORD_INSERTION,
            WordPerturbationType.WORD_REORDERING,
            WordPerturbationType.WORD_DUPLICATION,
            WordPerturbationType.STOP_WORD_REMOVAL
        }
        
        # Calculate average impact for semantic vs syntactic
        semantic_degradations = []
        syntactic_degradations = []
        
        for p_type, results_list in results['by_perturbation'].items():
            avg_degradation = np.mean([r['degradation'] for r in results_list])
            
            if p_type in semantic_perturbations:
                semantic_degradations.extend([r['degradation'] for r in results_list])
            elif p_type in syntactic_perturbations:
                syntactic_degradations.extend([r['degradation'] for r in results_list])
            
            # Identify high and low impact perturbations
            if avg_degradation > results['summary']['avg_degradation'] * 1.2:
                semantic_analysis['high_impact_perturbations'].append({
                    'type': p_type.value,
                    'avg_degradation': avg_degradation,
                    'impact': 'HIGH'
                })
            elif avg_degradation < results['summary']['avg_degradation'] * 0.8:
                semantic_analysis['low_impact_perturbations'].append({
                    'type': p_type.value,
                    'avg_degradation': avg_degradation,
                    'impact': 'LOW'
                })
        
        # Compare semantic vs syntactic
        if semantic_degradations and syntactic_degradations:
            semantic_analysis['semantic_vs_syntactic'] = {
                'semantic_avg': np.mean(semantic_degradations),
                'syntactic_avg': np.mean(syntactic_degradations),
                'semantic_more_robust': np.mean(semantic_degradations) < np.mean(syntactic_degradations)
            }
        
        # Generate recommendations
        self._generate_recommendations(semantic_analysis, results)
        
        return semantic_analysis
    
    # def _generate_recommendations(self, semantic_analysis: Dict, results: Dict):
    #     """Generate recommendations based on analysis"""
    #     recommendations = []
        
    #     # Check if model is more sensitive to semantic or syntactic changes
    #     if semantic_analysis['semantic_vs_syntactic']:
    #         if semantic_analysis['semantic_vs_syntactic']['semantic_more_robust']:
    #             recommendations.append(
    #                 "Model shows good semantic understanding. Consider improving "
    #                 "syntactic robustness through data augmentation with word reordering."
    #             )
    #         else:
    #             recommendations.append(
    #                 "Model is more sensitive to semantic changes. Consider using "
    #                 "semantic embeddings or synonym dictionaries in preprocessing."
    #             )
        
    #     # Specific high-impact perturbation recommendations
    #     for high_impact in semantic_analysis['high_impact_perturbations']:
    #         if high_impact['type'] == 'word_deletion':
    #             recommendations.append(
    #                 "High sensitivity to word deletion detected. Consider implementing "
    #                 "query expansion or using subword tokenization."
    #             )
    #         elif high_impact['type'] == 'abbreviation':
    #             recommendations.append(
    #                 "Model struggles with abbreviations. Implement abbreviation "
    #                 "expansion in preprocessing pipeline."
    #             )
        
    #     # Intensity-based recommendations
    #     summary = results['summary']
    #     intensity_degradations = [(i, stats['avg']) 
    #                             for i, stats in summary['by_intensity'].items()]
    #     intensity_degradations.sort(key=lambda x: x[0])
        
    #     if len(intensity_degradations) > 2:
    #         low_intensity_deg = intensity_degradations[0][1]
    #         high_intensity_deg = intensity_degradations[-1][1]
            
    #         if high_intensity_deg > 2 * low_intensity_deg:
    #             recommendations.append(
    #                 f"Degradation increases significantly with intensity "
    #                 f"({low_intensity_deg:.3f} to {high_intensity_deg:.3f}). "
    #                 "Consider implementing progressive error correction."
    #             )
        
    #     semantic_analysis['recommendations'] = recommendations
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive word-level test report"""
        report = []
        report.append("=== Word-Level IR Model Sensitivity Test Report ===\n")
        
        # Overall summary
        summary = results['summary']
        report.append(f"Overall Performance:")
        report.append(f"  Average Degradation: {summary['avg_degradation']:.3f}")
        report.append(f"  Std Dev: {summary['std_degradation']:.3f}")
        report.append(f"  Range: [{summary['min_degradation']:.3f}, {summary['max_degradation']:.3f}]")
        report.append("")
        
        # Precision metrics
        report.append("Retrieval Precision:")
        for k in [1, 3, 5, 10]:
            report.append(f"  P@{k}: {summary[f'avg_precision_at_{k}']:.3f}")
        report.append("")
        
        # By perturbation type
        report.append("Performance by Perturbation Type:")
        sorted_perturbations = sorted(
            summary['by_perturbation'].items(),
            key=lambda x: x[1]['avg'],
            reverse=True
        )
        
        for p_type, stats in sorted_perturbations:
            report.append(f"  {p_type}:")
            report.append(f"    Avg Degradation: {stats['avg']:.3f} (±{stats['std']:.3f})")
            report.append(f"    Avg Words Affected: {stats['avg_words_affected']:.1f}")
            report.append(f"    Test Count: {stats['count']}")
        report.append("")
        
        # Semantic analysis
        semantic = results['semantic_analysis']
        report.append("Semantic Analysis:")
        
        if semantic['semantic_vs_syntactic']:
            sem_avg = semantic['semantic_vs_syntactic']['semantic_avg']
            syn_avg = semantic['semantic_vs_syntactic']['syntactic_avg']
            report.append(f"  Semantic Changes: {sem_avg:.3f} avg degradation")
            report.append(f"  Syntactic Changes: {syn_avg:.3f} avg degradation")
            report.append(f"  More Robust to: {'Semantic' if semantic['semantic_vs_syntactic']['semantic_more_robust'] else 'Syntactic'} changes")
        report.append("")
        
        # High impact perturbations
        if semantic['high_impact_perturbations']:
            report.append("High Impact Perturbations:")
            for pert in semantic['high_impact_perturbations']:
                report.append(f"  - {pert['type']}: {pert['avg_degradation']:.3f} degradation")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        for i, rec in enumerate(semantic['recommendations'], 1):
            report.append(f"  {i}. {rec}")
        
        return "\n".join(report)
