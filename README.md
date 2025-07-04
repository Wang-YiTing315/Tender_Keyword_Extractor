# Tender_Keyword_Extractor
# Keyword & Product Phrase Extraction from Tender Data

## Project Overview

This project is designed to automatically extract high-frequency keywords and product category phrases from massive international tender datasets, specifically analyzing the `Summary` and `Description` fields. The goal is to help users quickly understand what products are being procured in the global market. The project has evolved through several versions, each improving the accuracy and intelligence of phrase extraction.

## Main Features

- **Text Cleaning & Tokenization**: Preprocesses tender data, including lowercasing, tokenization, and lemmatization.
- **Meaningful Phrase Extraction**: Uses POS tagging to identify and extract high-frequency phrases with structures like “JJ+NN” (adjective + noun) and “NN+NN” (noun + noun).
- **TF-IDF Scoring**: Calculates TF-IDF scores for all extracted phrases to identify the most representative keywords and phrases.
- **Product Category Recognition**: Integrates with large language model APIs (e.g., deepseek-chat) to automatically determine whether a phrase is a product category and can generate concise product descriptions.
- **Result Export**: Supports exporting results to Excel files for further analysis and visualization.

## Version Evolution

### v1.5
- Implements basic text cleaning, tokenization, phrase extraction, and TF-IDF calculation.
- Allows custom minimum phrase frequency; results can be exported to Excel.
- **Limitation**: Only extracts frequent phrases, cannot distinguish product categories, and lacks semantic understanding.

### v2.5
- Adds concurrent calls to LLM APIs (deepseek-chat) to automatically determine if a phrase is a product category.
- Supports multithreading to speed up product category filtering.
- **Limitation**: Product category detection only returns “yes/no”, without further description.

### v3.5
- Builds on v2.5 by generating an automated description for each product category phrase, using real tender texts to output industry-standard, concise product definitions.
- Improves API robustness (retry mechanism, timeout handling) for better stability.
- **Limitation**: Relies on external LLM APIs, subject to rate limits; some product descriptions may lack precision due to limited data samples.

## Usage

1. Prepare an Excel file containing `Summary` and `Description` columns.
2. Run the corresponding Python script (e.g., `phrase_extraction_v3.5.py`) and follow the prompts to input the file name and minimum phrase frequency.
3. Wait for the program to process; the final Excel file with keywords/product categories and descriptions will be generated in the current directory.

## Limitations & Suggestions for Improvement

### Limitations
- **API Dependency**: Product category recognition and description generation rely on external LLM services, which are subject to rate limits and stability issues.
- **Language Support**: Currently supports only English data; cannot directly process multilingual or mixed-language corpora.
- **Contextual Understanding**: Phrase extraction is based on fixed POS patterns, which may miss valuable phrases in complex contexts.
- **Description Consistency**: Some product descriptions may be inaccurate due to insufficient or noisy data samples.

### Suggestions for Improvement
- **Local Model Alternatives**: Consider integrating locally deployed NLP models to reduce reliance on external APIs.
- **Multilingual Support**: Extend tokenization and POS tagging modules to support Chinese and other languages.
- **Contextual Enhancement**: Combine dependency parsing and entity recognition to improve extraction of complex phrases and product categories.
- **Description Generation Optimization**: Use smarter summarization algorithms or fine-tuned LLMs to improve the accuracy and industry relevance of product descriptions.
- **Visualization & Interaction**: Develop a frontend interface for visualizing keyword trends, product distributions, and more.

## Acknowledgements

Thanks to all open-source NLP tools and LLM API providers. Contributions and suggestions are welcome—feel free to open issues or PRs to help improve this project!
