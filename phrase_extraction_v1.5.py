import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter
import warnings
import math
warnings.filterwarnings('ignore')

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class PhraseExtractor: #词组提取器
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer() #词形还原
        self.stop_words = set(stopwords.words('english')) #停用词
        
    def clean_text(self, text): 
        """清理文本，只转小写，不去掉标点"""
        if pd.isna(text):
            return ""
        text = str(text).lower() # 只转小写，不去标点
        return text
    
    def tokenize_text(self, text): 
        """分词"""
        if not text:
            return []
        tokens = word_tokenize(text) #分词
        return tokens
    
    def standardize_words(self, tokens):
        """标准化单词"""
        standardized = []
        for token in tokens:
            if len(token) > 1:
                lemma = self.lemmatizer.lemmatize(token) #词形还原
                standardized.append(lemma) #将词形还原后的结果添加到standardized列表中
        return standardized
    
    def extract_meaningful_phrases(self, tokens): 
        """提取有意义的词组（JJ+NN或NN+NN），不跨标点，组词组时再做词形还原"""
        if not tokens or len(tokens) < 2:
            return []
        
        pos_tags = pos_tag(tokens)
        meaningful_phrases = []
        
        for i in range(len(pos_tags) - 1):
            word1, pos1 = pos_tags[i]
            word2, pos2 = pos_tags[i + 1]
            
            # 跳过标点
            if not word1.isalpha() or not word2.isalpha():
                continue
            
            # 词形还原
            word1_lemma = self.lemmatizer.lemmatize(word1.lower())
            word2_lemma = self.lemmatizer.lemmatize(word2.lower())
            
            # 检查是否为JJ+NN或NN+NN组合
            if ((pos1.startswith('JJ') and pos2.startswith('NN')) or 
                (pos1.startswith('NN') and pos2.startswith('NN'))):
                if (word1_lemma not in self.stop_words and 
                    word2_lemma not in self.stop_words and
                    len(word1_lemma) > 2 and 
                    len(word2_lemma) > 2 and
                    word1_lemma != word2_lemma):
                    phrase = f"{word1_lemma} {word2_lemma}"
                    meaningful_phrases.append(phrase)
        
        return meaningful_phrases
    
    def calculate_tfidf_scores(self, doc_phrases_list, phrase_freq):
        """计算TF-IDF分数 - 基于我们自己提取的词组"""
        # 计算IDF
        total_docs = len(doc_phrases_list)
        phrase_doc_freq = Counter()  # 记录每个词组在多少文档中出现
        
        # 统计每个词组出现在多少文档中
        for doc_phrases in doc_phrases_list:
            # 使用set去重，每个文档中的词组只计算一次
            unique_phrases = set(doc_phrases)
            for phrase in unique_phrases:
                phrase_doc_freq[phrase] += 1
        
        # 计算每个词组的TF-IDF分数
        phrase_scores = {}
        total_phrase_count = sum(phrase_freq.values())  # 新增：所有词组出现的总次数
        for phrase, total_freq in phrase_freq.items():
            # 计算IDF：log(总文档数 / (出现该词组的文档数 + 1))
            doc_freq = phrase_doc_freq.get(phrase, 0)  #出现该词组的文档数
            idf = math.log(total_docs / (doc_freq + 1)) #log(总文档数 / (出现该词组的文档数 + 1))
            
            # 计算TF：该词组出现的次数 / 所有词组出现的总次数
            tf = total_freq / total_phrase_count
            
            # 计算TF-IDF
            tfidf_score = tf * idf
            phrase_scores[phrase] = tfidf_score
        return phrase_scores
    
    def extract_phrases(self, df, summary_col='Summary', description_col='Description', min_freq=20):
        """主要的词组提取函数"""
        print("开始词组提取...")
        
        # 1. 合并Summary和Description
        print("1. 合并Summary和Description...")
        combined_texts = []
        for idx, row in df.iterrows():
            summary = str(row.get(summary_col, '')) if pd.notna(row.get(summary_col)) else ''
            description = str(row.get(description_col, '')) if pd.notna(row.get(description_col)) else ''
            combined = f"{summary} {description}".strip()
            combined_texts.append(combined)
        
        # 2. 分词和提取词组
        print("2. 分词和提取词组...")
        all_phrases = []
        doc_phrases_list = []  # 新增：存储每条数据的词组列表
        
        for text in combined_texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_text(cleaned_text)
            phrases = self.extract_meaningful_phrases(tokens)
            all_phrases.extend(phrases)
            doc_phrases_list.append(phrases)  # 新增：保存这条数据的词组列表
        
        # 3. 计算词组频率
        print("3. 计算词组频率...")
        phrase_freq = Counter(all_phrases)
        total_phrases = len(all_phrases)
        
        # 4. 计算TF-IDF分数
        print("4. 计算TF-IDF分数...")
        tfidf_scores = self.calculate_tfidf_scores(doc_phrases_list, phrase_freq)
        
        # 5. 生成结果
        print("5. 生成结果...")
        results = []
        for phrase, freq in phrase_freq.items():
            if freq >= min_freq: # 用输入的最小频率
                tfidf_score = tfidf_scores.get(phrase, 0)
                results.append({
                    'phrase': phrase,
                    'frequency': freq,
                    'tfidf_score': tfidf_score
                })
        results.sort(key=lambda x: x['tfidf_score'], reverse=True)
        return results
    
    def save_results(self, results, output_file='phrase_results.xlsx'): #保存结果到Excel
        """保存结果到Excel"""
        df_results = pd.DataFrame(results)
        df_results.to_excel(output_file, index=False)
        print(f"结果已保存到: {output_file}")
        
        # 打印前20个词组
        print("\n前20个词组:")
        for i, result in enumerate(results[:20]): #遍历results列表中的前20个元素
            print(f"{i+1}. {result['phrase']} (频率: {result['frequency']}, TF-IDF: {result['tfidf_score']:.4f})")

def main():
    try:
        # 询问要分析的Excel文件名
        excel_file = input("请输入要分析的Excel文件名（如400+原数据.xlsx）：")
        if not excel_file.strip():
            print("文件名不能为空！")
            return
        # 询问最小频率
        min_freq_input = input("请输入出现在表格中词组的最小频率（默认为20）：")
        min_freq = 20
        if min_freq_input.strip():
            try:
                min_freq = int(min_freq_input)
            except ValueError:
                print("输入无效，使用默认值20。")
                min_freq = 20
        print("读取Excel文件...")
        df = pd.read_excel(excel_file)
        print(f"成功读取数据，共{len(df)}行")
        print(f"列名: {list(df.columns)}")
        
        # 创建词组提取器
        extractor = PhraseExtractor()
        
        # 提取词组
        results = extractor.extract_phrases(df, min_freq=min_freq)
        
        # 保存结果
        extractor.save_results(results)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保Excel文件存在且包含Summary和Description列")

if __name__ == "__main__":
    main() 