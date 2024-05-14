#任务1，统计分析每个论文类别在不同时期的热门关键词，分析arXiv论文常见的关键词发展趋势，并进行统计分析可视化：
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook

output_notebook()

# 数据加载
file_path = 'path'
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 转换为DataFrame
df = pd.DataFrame(data)

# TF-IDF和LDA
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['abstract'])
lda = LDA(n_components=10, learning_method='batch', max_iter=30, random_state=0)
lda_matrix = lda.fit_transform(tfidf_matrix)

# 主题词提取
feature_names = tfidf.get_feature_names_out()
topic_keywords = {}
for topic_idx, topic in enumerate(lda.components_):
    topic_keywords[f"Topic {topic_idx}"] = [feature_names[i] for i in topic.argsort()[-10:]]

# 主题趋势
df['year'] = df['versions'].apply(lambda x: x[0]['created'][:4])
topic_trends = pd.DataFrame(lda_matrix, columns=topic_keywords.keys())
topic_trends['year'] = df['year']

# 按年份和主题聚合
yearly_trends = topic_trends.groupby('year').mean().stack().reset_index()
yearly_trends.columns = ['Year', 'Topic', 'Score']

# 可视化
p = figure(x_range=yearly_trends['Year'].unique(), title="Topic Trends Over Years", plot_height=300, plot_width=800)
p.vbar(x='Year', top='Score', source=ColumnDataSource(yearly_trends), width=0.9)
show(p)



from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 打印主题词
for topic, keywords in topic_keywords.items():
    print(f"Topic {topic}: {', '.join(keywords)}")

# 可视化主题词
for topic, keywords in topic_keywords.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Topic {topic} Keywords')
    plt.axis('off')
    plt.show()


import matplotlib.pyplot as plt

# 按年份和主题聚合
yearly_trends = topic_trends.groupby(['year']).mean()

# 可视化主题趋势
plt.figure(figsize=(10, 6))
for topic in yearly_trends.columns:
    plt.plot(yearly_trends.index, yearly_trends[topic], label=topic)
plt.title('Topic Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()


import seaborn as sns

# 绘制主题分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(data=topic_trends.drop(columns='year'), bins=20, kde=True, alpha=0.5)
plt.title('Topic Distribution Histogram')
plt.xlabel('Topic Score')
plt.ylabel('Frequency')
plt.show()


#task2：
import pandas as pd
import json
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')

# 数据加载
file_path = 'path'
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 转换为DataFrame
df = pd.DataFrame(data)

# 函数计算句子长度和情感
def analyze_abstract(text):
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    average_length = sum(sentence_lengths) / len(sentences) if sentences else 0
    sentiment = TextBlob(text).sentiment.polarity
    articles = sum(1 for word in text.split() if word.lower() in ['the', 'a', 'an'])
    return average_length, sentiment, articles

# 应用函数
df['analysis'] = df['abstract'].apply(analyze_abstract)
df[['average_sentence_length', 'sentiment', 'article_count']] = df['analysis'].apply(pd.Series)

# 绘制可视化
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
sns.boxplot(x='categories', y='average_sentence_length', data=df, ax=axes[0])
axes[0].set_title('Average Sentence Length by Category')
sns.boxplot(x='categories', y='sentiment', data=df, ax=axes[1])
axes[1].set_title('Sentiment by Category')
sns.boxplot(x='categories', y='article_count', data=df, ax=axes[2])
axes[2].set_title('Article Count by Category')
plt.tight_layout()
plt.show()



#task3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置seaborn样式
sns.set(style="whitegrid")

# 读取JSON文件数据
path = "path"
data = []
with open(path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc='Loading data'):
        data.append(json.loads(line))

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 只保留我们需要的列
df = df[['abstract', 'update_date']]

# 关键词提取
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(tqdm(df['abstract'], desc='Extracting keywords'))
keywords = vectorizer.get_feature_names_out()

# 将日期转换为月份
df['month'] = pd.to_datetime(df['update_date'], errors='coerce').dt.to_period('M')

# 检查是否有 NaT 值并移除
if df['month'].isnull().any():
    print("There are some invalid dates which will be removed.")
    df = df.dropna(subset=['month'])

# 构建关键词月份矩阵
keyword_trends_month = pd.DataFrame(X.toarray(), columns=keywords)
keyword_trends_month['month'] = df['month'].values
keyword_trends_month = keyword_trends_month.groupby('month').sum()

# 选择前10个最常见的关键词
top_keywords_month = keyword_trends_month.sum().sort_values(ascending=False).head(10).index

# 提取前10个关键词的趋势数据
top_keyword_trends_month = keyword_trends_month[top_keywords_month]

# 绘制前10个关键词的趋势图（按月份）
plt.figure(figsize=(12, 8))
for keyword in top_keywords_month:
    plt.plot(top_keyword_trends_month.index.astype(str), top_keyword_trends_month[keyword], label=keyword)
plt.xlabel('Month')
plt.ylabel('Number of Papers')
plt.title('Top 10 Keywords Trend Over Months')
plt.legend()
plt.show()

# 绘制堆积面积图（按月份）
top_keyword_trends_month.plot.area(figsize=(12, 8), alpha=0.5)
plt.xlabel('Month')
plt.ylabel('Number of Papers')
plt.title('Top 10 Keywords Trend Over Months (Stacked)')
plt.show()
