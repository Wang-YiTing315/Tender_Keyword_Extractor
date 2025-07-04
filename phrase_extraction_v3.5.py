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
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
        phrase_doc_freq = Counter()  # 记录每个词组在多少条数据中出现
        
        # 统计每个词组出现在多少条数据中
        for doc_phrases in doc_phrases_list:
            unique_phrases = set(doc_phrases)
            for phrase in unique_phrases:
                phrase_doc_freq[phrase] += 1
        
        # 计算每个词组的TF-IDF分数
        phrase_scores = {}
        total_phrase_count = sum(phrase_freq.values())  # 所有词组出现的总次数
        for phrase, total_freq in phrase_freq.items():
            # 计算IDF：log(总数据条数 / (出现该词组的数据条数 + 1))
            doc_freq = phrase_doc_freq.get(phrase, 0)
            idf = math.log(total_docs / (doc_freq + 1))
            # 计算TF：该词组出现的次数 / 所有词组出现的总次数
            tf = total_freq / total_phrase_count
            # 计算TF-IDF
            tfidf_score = tf * idf
            phrase_scores[phrase] = tfidf_score
        return phrase_scores
    
    def extract_phrases(self, df, summary_col='Summary', description_col='Description', min_freq=2):
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

# deepseek-chat 判断产品词组函数
def is_product_phrase(phrase, api_key, max_retries=3, retry_delay=3):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = """
    #### 角色
    你是产品词组专家

    #### 任务
    判断英文词组是否为产品类别词组

    #### 举例
    产品类别词组包括但不限于：solar panel、power cable 等
    
    #### 输出
    你只需要回答是或否
    """
    user_prompt = f"英文词组为 \"{phrase}\" 。"
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()    # 调用deepseek-chat判断词组是否为产品词组，超时时间15秒
            answer = result['choices'][0]['message']['content'].strip()
            return "是" in answer
        except Exception as e:
            print(f"API 调用失败: {e}，正在重试({attempt+1}/{max_retries})...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return False

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
        
        # 新增：并发调用 deepseek-chat 判断产品词组
        api_key = "your deepseek apikey"
        product_phrases = []
        print("正在并发判断哪些词组为产品类别词组...")
        def check_and_collect(result):
            phrase = result['phrase']
            if is_product_phrase(phrase, api_key):
                return result
            return None
        with ThreadPoolExecutor(max_workers=5) as executor:  # 减少到5个线程，避免API压力过大
            future_to_result = {executor.submit(check_and_collect, result): result for result in results}
            for future in tqdm(as_completed(future_to_result), total=len(results), desc="产品词组筛选"):
                res = future.result()
                if res:
                    product_phrases.append(res)
        if product_phrases:
            df_products = pd.DataFrame(product_phrases)

            # 新增：为每个产品词组生成一句话描述
            print("正在为每个产品类别词组生成一句话描述...")
            df_origin = pd.read_excel(excel_file)
            descriptions = []
            def get_product_description(phrase, texts, api_key, max_retries=3, retry_delay=3):
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # 随机抽样20条数据，避免数据量过大
                if len(texts) > 20:
                    import random
                    sampled_texts = random.sample(texts, 20)
                else:
                    sampled_texts = texts
                
                combined_text = "\n".join(sampled_texts)
                system_prompt = """
                #### 角色
                你是一位数据分析师/市场分析师。

                #### 指令
                梳理任务的背景、进展和想法，对于输入的产品类别词组，输出一段描述。

                #### 背景
                你正帮助客户分析全球国际招投标市场中的采购类型，目的是撰写一份关于全球招投标市场的市场情况分析报告。报告的核心目标是明确全球市场中哪些产品被采购，涵盖其产品类别、具体描述、规格、使用场景等信息。此外，报告还需要呈现不同产品类别的采购需求随时间变化的趋势，并通过数据可视化展示。

                #### 进展
                你已经成功搜集了大量的招投标信息，其中每条信息都包含了一个具体的描述。这些数据尚未分类。为了进行分析，你采用了传统的 tf-idf 方法，从这些招投标信息中提取出了高频的产品名称。这些名称代表了潜在的高频采购类型，但单独的产品名称并没有提供足够的具体含义，因此需要进一步丰富这些描述。

                #### 想法
                你的想法是，提取每个高频的产品名称后，找到至少 10 条包含该产品的招投标信息。然后，对这些信息进行分析，从而生成更为丰富的产品类别描述。这些描述应包括但不限于采购产品的功能、潜在的规格、类型、使用场景等具体信息，进一步帮助定义这些采购类型的具体特征。
                    
                #### 输入
                以下是关于某产品类别词组 \"{phrase}\" 的多条招投标信息：\n{combined_text}\n 

                #### 输出
                请严格遵循以下要求：
                1. 输出格式为：产品描述：{description} 
                2. 产品描述需要包括产品类别词组，并进一步丰富产品描述，包括产品功能、潜在的规格、类型、使用场景等具体信息
                3. 产品描述需要简洁明了，不要超过150字
                4. 产品描述需要符合行业规范，不要出现敏感词汇
                5. 产品描述使用中文输出
                举例：
                产品类别词组：control panel
                产品描述：Control panel（控制面板）是一种用于监控和管理设备运行的电子或电气装置，常见于电梯、消防系统、水泵、通风系统等场景。其功能包括设备启停、参数调节、状态显示及故障报警等。控制面板通常配备按钮、显示屏、继电器、传感器等组件，并可能集成自动化控制模块。根据应用需求，控制面板可分为工业级、楼宇自动化、医疗设备专用等类型，具有防水、防尘或防爆等特性。
                    """
                user_prompt = f"以下是关于某产品类别词组 \"{phrase}\" 的多条招投标信息：\n{combined_text}\n "
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                for attempt in range(max_retries):
                    try:
                        response = requests.post(url, headers=headers, json=data, timeout=40)
                        response.raise_for_status()
                        result = response.json()  # 调用deepseek-chat生成产品描述，超时时间60秒
                        answer = result['choices'][0]['message']['content'].strip()
                        return answer
                    except Exception as e:
                        print(f"API 调用失败: {e}，正在重试({attempt+1}/{max_retries})...")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                return ""
            
            # 准备产品描述生成的数据
            def prepare_product_data(product):
                phrase = product['phrase']
                mask = df_origin.apply(
                    lambda row: phrase in str(row.get('Summary', '')) or phrase in str(row.get('Description', '')),
                    axis=1
                )
                matched_rows = df_origin[mask]
                related_texts = []
                for _, row in matched_rows.iterrows():
                    summary = str(row.get('Summary', ''))
                    description = str(row.get('Description', ''))
                    related_texts.append(f"{summary} {description}".strip())
                return phrase, related_texts
            
            # 并发生成产品描述
            def generate_description_with_data(product_data):
                phrase, related_texts = product_data
                desc = get_product_description(phrase, related_texts, api_key)
                return phrase, desc
            
            # 准备所有产品数据
            product_data_list = [prepare_product_data(product) for product in product_phrases]
            
            # 并发处理产品描述生成
            descriptions_dict = {}
            with ThreadPoolExecutor(max_workers=5) as executor:  # 使用5个线程并发处理
                future_to_phrase = {executor.submit(generate_description_with_data, data): data[0] for data in product_data_list}
                for future in tqdm(as_completed(future_to_phrase), total=len(product_phrases), desc="生成产品描述"):
                    phrase, desc = future.result()
                    descriptions_dict[phrase] = desc
            
            # 按原始顺序添加描述
            for product in product_phrases:
                phrase = product['phrase']
                descriptions.append(descriptions_dict.get(phrase, ""))
            df_products['产品描述'] = descriptions
            df_products.to_excel("产品词组数据.xlsx", index=False)
            print("产品类别词组及描述已保存到: 产品词组数据.xlsx")
        else:
            print("未识别到产品类别词组。")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保Excel文件存在且包含Summary和Description列")

if __name__ == "__main__":
    main() 
