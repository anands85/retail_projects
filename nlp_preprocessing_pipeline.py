"""
Complete NLP Text Preprocessing Pipeline
Optimized for best performance and accuracy
"""

import re
import string
import unicodedata
from typing import List, Optional, Union, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Core NLP libraries
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import wordnet

# Text cleaning and correction
import contractions
from autocorrect import Speller
import ftfy  # fixes text encoding issues

# Performance and utilities
import pandas as pd
import numpy as np
from collections import Counter
import time
from functools import lru_cache

class TextPreprocessor:
    """
    Complete text preprocessing pipeline with configurable steps
    Optimized for performance and accuracy
    """
    
    def __init__(self, 
                 language: str = 'en',
                 spell_check: bool = True,
                 lemmatize: bool = True,
                 remove_stopwords: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True,
                 min_word_length: int = 2,
                 max_word_length: int = 50,
                 preserve_entities: bool = False,
                 custom_stopwords: Optional[List[str]] = None,
                 custom_replacements: Optional[Dict[str, str]] = None):
        """
        Initialize the text preprocessor with configurable options
        
        Args:
            language: Language code (default: 'en')
            spell_check: Enable spell checking
            lemmatize: Enable lemmatization
            remove_stopwords: Remove stop words
            remove_punctuation: Remove punctuation marks
            lowercase: Convert to lowercase
            remove_numbers: Remove numeric characters
            remove_extra_whitespace: Clean up extra whitespace
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            preserve_entities: Preserve named entities
            custom_stopwords: Additional stopwords to remove
            custom_replacements: Custom text replacements
        """
        self.language = language
        self.spell_check = spell_check
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.preserve_entities = preserve_entities
        self.custom_stopwords = custom_stopwords or []
        self.custom_replacements = custom_replacements or {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize NLP components and download required data"""
        print("Initializing NLP components...")
        
        # Download required NLTK data
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'omw-1.4'
        ]
        
        for resource in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
        
        # Initialize spaCy model for advanced processing
        try:
            if self.language == 'en':
                self.nlp = spacy.load('en_core_web_sm')
            else:
                print(f"Warning: spaCy model for {self.language} not available, using basic processing")
                self.nlp = None
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize other components
        if self.spell_check:
            self.speller = Speller(lang=self.language, fast=True)
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        # Get stopwords
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english' if self.language == 'en' else self.language))
                self.stop_words.update(self.custom_stopwords)
            except:
                self.stop_words = set(self.custom_stopwords)
        
        print("Initialization complete!")
    
    @lru_cache(maxsize=10000)
    def _get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character used by WordNetLemmatizer"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def fix_encoding(self, text: str) -> str:
        """Fix text encoding issues"""
        return ftfy.fix_text(text)
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKD', text)
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions (e.g., don't -> do not)"""
        try:
            return contractions.fix(text)
        except:
            # Fallback manual contractions
            contraction_dict = {
                "ain't": "am not", "aren't": "are not", "can't": "cannot",
                "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                "haven't": "have not", "he'd": "he would", "he'll": "he will",
                "he's": "he is", "i'd": "i would", "i'll": "i will",
                "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'll": "it will", "it's": "it is",
                "let's": "let us", "shouldn't": "should not", "that's": "that is",
                "there's": "there is", "they'd": "they would", "they'll": "they will",
                "they're": "they are", "they've": "they have", "we'd": "we would",
                "we're": "we are", "we've": "we have", "weren't": "were not",
                "what's": "what is", "where's": "where is", "who's": "who is",
                "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will", "you're": "you are", "you've": "you have"
            }
            
            words = text.split()
            for i, word in enumerate(words):
                words[i] = contraction_dict.get(word.lower(), word)
            return ' '.join(words)
    
    def remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        return re.sub(r'<[^>]+>', '', text)
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters and excessive punctuation"""
        # Remove excessive punctuation (more than 2 consecutive)
        text = re.sub(r'[^\w\s]{3,}', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        return text
    
    def spell_correct(self, text: str) -> str:
        """Apply spell correction"""
        if not self.spell_check:
            return text
        
        try:
            # Only correct words that are likely misspelled (not proper nouns, etc.)
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Skip if word is too short, has numbers, or is capitalized (might be proper noun)
                if len(word) < 3 or any(char.isdigit() for char in word) or word[0].isupper():
                    corrected_words.append(word)
                else:
                    corrected = self.speller(word.lower())
                    corrected_words.append(corrected if corrected != word.lower() else word)
            
            return ' '.join(corrected_words)
        except:
            return text
    
    def apply_custom_replacements(self, text: str) -> str:
        """Apply custom text replacements"""
        for old, new in self.custom_replacements.items():
            text = text.replace(old, new)
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        if not self.lemmatize:
            return tokens
        
        if self.nlp:
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc if not token.is_space]
        else:
            return [self.lemmatizer.lemmatize(token, self._get_wordnet_pos(token)) for token in tokens]
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on length and content"""
        filtered = []
        for token in tokens:
            # Skip if token is too short or too long
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue
            
            # Remove punctuation if specified
            if self.remove_punctuation and token in string.punctuation:
                continue
            
            # Remove numbers if specified
            if self.remove_numbers and token.isdigit():
                continue
            
            filtered.append(token)
        
        return filtered
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {"PERSON": [], "ORG": [], "GPE": [], "MISC": []}
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    entities[ent.label_].append(ent.text)
                else:
                    entities["MISC"].append(ent.text)
        else:
            # Fallback using NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    label = chunk.label()
                    if label in entities:
                        entities[label].append(entity)
                    else:
                        entities["MISC"].append(entity)
        
        return entities
    
    def preprocess_single(self, text: str) -> Union[str, Dict[str, Any]]:
        """
        Preprocess a single text document
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text or dictionary with text and metadata
        """
        if not isinstance(text, str):
            return ""
        
        original_text = text
        metadata = {}
        
        # Step 1: Fix encoding issues
        text = self.fix_encoding(text)
        
        # Step 2: Normalize unicode
        text = self.normalize_unicode(text)
        
        # Step 3: Extract entities if needed (before lowercasing)
        if self.preserve_entities:
            metadata['entities'] = self.extract_entities(text)
        
        # Step 4: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 5: Remove URLs and emails
        text = self.remove_urls_emails(text)
        
        # Step 6: Expand contractions
        text = self.expand_contractions(text)
        
        # Step 7: Apply custom replacements
        text = self.apply_custom_replacements(text)
        
        # Step 8: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 9: Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Step 10: Spell correction
        text = self.spell_correct(text)
        
        # Step 11: Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 12: Tokenization
        tokens = self.tokenize_text(text)
        
        # Step 13: Remove stopwords
        tokens = self.remove_stopwords_func(tokens)
        
        # Step 14: Lemmatization
        tokens = self.lemmatize_tokens(tokens)
        
        # Step 15: Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Return processed text
        processed_text = ' '.join(tokens)
        
        if self.preserve_entities:
            return {
                'original_text': original_text,
                'processed_text': processed_text,
                'tokens': tokens,
                'metadata': metadata,
                'word_count': len(tokens),
                'char_count': len(processed_text)
            }
        else:
            return processed_text
    
    def preprocess_batch(self, texts: List[str], 
                        show_progress: bool = True,
                        n_jobs: int = 1) -> List[Union[str, Dict[str, Any]]]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            n_jobs: Number of parallel jobs (currently supports 1)
            
        Returns:
            List of preprocessed texts
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and i % 100 == 0:
                print(f"Processing: {i}/{total} ({i/total*100:.1f}%)")
            
            result = self.preprocess_single(text)
            results.append(result)
        
        if show_progress:
            print(f"Processing complete: {total}/{total} (100.0%)")
        
        return results
    
    def get_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Get statistics about the text corpus"""
        if not texts:
            return {}
        
        processed = self.preprocess_batch(texts, show_progress=False)
        
        if isinstance(processed[0], dict):
            word_counts = [item['word_count'] for item in processed]
            char_counts = [item['char_count'] for item in processed]
            all_tokens = [token for item in processed for token in item['tokens']]
        else:
            word_counts = [len(text.split()) for text in processed]
            char_counts = [len(text) for text in processed]
            all_tokens = [token for text in processed for token in text.split()]
        
        vocab = Counter(all_tokens)
        
        return {
            'total_documents': len(texts),
            'avg_words_per_doc': np.mean(word_counts),
            'avg_chars_per_doc': np.mean(char_counts),
            'vocabulary_size': len(vocab),
            'most_common_words': vocab.most_common(10),
            'total_words': sum(word_counts),
            'total_chars': sum(char_counts)
        }


def create_preprocessing_pipeline(config: Optional[Dict[str, Any]] = None) -> TextPreprocessor:
    """
    Factory function to create a preprocessing pipeline with predefined configurations
    
    Args:
        config: Configuration dictionary or predefined config name
        
    Returns:
        Configured TextPreprocessor instance
    """
    
    # Predefined configurations
    configs = {
        'basic': {
            'spell_check': False,
            'lemmatize': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
            'lowercase': True
        },
        'advanced': {
            'spell_check': True,
            'lemmatize': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
            'lowercase': True,
            'preserve_entities': True
        },
        'minimal': {
            'spell_check': False,
            'lemmatize': False,
            'remove_stopwords': False,
            'remove_punctuation': False,
            'lowercase': True,
            'remove_extra_whitespace': True
        },
        'social_media': {
            'spell_check': True,
            'lemmatize': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
            'lowercase': True,
            'custom_replacements': {
                'u': 'you', 'ur': 'your', 'r': 'are', 'n': 'and',
                'luv': 'love', 'gud': 'good', 'gr8': 'great'
            }
        }
    }
    
    if isinstance(config, str) and config in configs:
        return TextPreprocessor(**configs[config])
    elif isinstance(config, dict):
        return TextPreprocessor(**config)
    else:
        return TextPreprocessor()


# Example usage and testing
if __name__ == "__main__":
    # Sample texts for testing
    sample_texts = [
        "Hello! This is a sample text with some speling errors and contractions like don't and won't.",
        "Visit https://example.com for more info. Email us at test@email.com!",
        "The U.S.A. is a country in North America. Dr. Smith works at OpenAI Inc.",
        "    Extra   whitespace    and    HTML <b>tags</b> should be cleaned!!!   ",
        "Social media text: ur gr8! luv u 2 ❤️ #awesome"
    ]
    
    print("=== Text Preprocessing Pipeline Demo ===\n")
    
    # Test different configurations
    configs_to_test = ['basic', 'advanced', 'minimal', 'social_media']
    
    for config_name in configs_to_test:
        print(f"--- {config_name.upper()} Configuration ---")
        preprocessor = create_preprocessing_pipeline(config_name)
        
        for i, text in enumerate(sample_texts[:2]):  # Test first 2 texts
            result = preprocessor.preprocess_single(text)
            print(f"Original: {text}")
            if isinstance(result, dict):
                print(f"Processed: {result['processed_text']}")
                print(f"Tokens: {result['tokens']}")
            else:
                print(f"Processed: {result}")
            print()
        
        print("-" * 50)
    
    # Performance test
    print("\n=== Performance Test ===")
    preprocessor = create_preprocessing_pipeline('advanced')
    
    start_time = time.time()
    results = preprocessor.preprocess_batch(sample_texts * 20)  # 100 texts
    end_time = time.time()
    
    print(f"Processed {len(results)} texts in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(results):.4f} seconds")
    
    # Get corpus statistics
    print("\n=== Corpus Statistics ===")
    stats = preprocessor.get_stats(sample_texts)
    for key, value in stats.items():
        print(f"{key}: {value}")
