import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 1. 数据集准备
try:
    data = pd.read_excel("产品评价.xlsx")
except FileNotFoundError:
    print("错误：'产品评价.xlsx' 文件未找到。请确保文件在当前工作目录中。")
    exit()

# 检查数据列名
print("数据集列名:", data.columns)
text_column_name = '评论'
label_column_name = '评价'

if text_column_name not in data.columns or label_column_name not in data.columns:
    print(f"错误：数据集中未找到预期的列 '{text_column_name}' 或 '{label_column_name}'。")
    print(f"请确保Excel文件包含这两列，或者在脚本中更新列名。")
    exit()

# 可视化数据集分布
plt.figure(figsize=(6, 4))
sns.countplot(x=label_column_name, data=data)
plt.title('产品评价分布 (0:差评, 1:好评)')
plt.xlabel('评价类别')
plt.ylabel('数量')
plt.savefig('dataset_distribution.png')
print("\n数据集分布图已保存为 dataset_distribution.png")

# 2. 中文分词
data[text_column_name] = data[text_column_name].astype(str)
data['分词评论'] = data[text_column_name].apply(lambda x: " ".join(jieba.cut(x)))

# 3. 特征提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['分词评论'])
y = data[label_column_name]

# 4. 模型训练 - 数据拆分
if len(y.unique()) < 2:
    print("错误：目标变量 '评价' 中的类别少于2个。无法进行分层抽样或训练分类器。")
    print(f"目标变量的唯一值: {y.unique()}")
    exit()
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 模型训练 - 训练MLP模型
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=10)
print("\n开始训练神经网络模型...")
mlp.fit(X_train, y_train)
print("模型训练完成。")

# 5. 模型预测
y_pred = mlp.predict(X_test)

# 5. 模型评估
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['差评 (0)', '好评 (1)'])
cm = confusion_matrix(y_test, y_pred)

# 6. 结果分析
print(f"\n模型在测试集上的准确率: {accuracy:.4f}")
print("\n详细分类报告:")
print(report)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['差评 (0)', '好评 (1)'], yticklabels=['差评 (0)', '好评 (1)'])
plt.xlabel('预测标签')
plt.ylabel('实际标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png')
print("\n混淆矩阵图已保存为 confusion_matrix.png")

# print("\n部分预测结果:")
# results_df = pd.DataFrame({'实际评价': y_test, '预测评价': y_pred})
# print(results_df.head())
