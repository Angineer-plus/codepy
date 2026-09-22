# 导入所有依赖库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据加载与预处理 =====================
# 加载红酒数据集（sklearn自带，13个理化特征，3类品质等级）
wine = datasets.load_wine()
X = wine.data    # 13个特征：酒精、苹果酸、灰分、镁含量等
y = wine.target  # 标签：0/1/2 对应3个品质等级
feature_names = wine.feature_names
target_names = wine.target_names

# 拆分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 数据标准化（消除不同特征的量纲差异）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 2. 实验1：4种算法分类效果对比 =====================
models = {
    "K近邻(KNN)": KNeighborsClassifier(n_neighbors=5),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "高斯朴素贝叶斯": GaussianNB()
}

# 训练并计算每个模型的准确率
accuracy_results = {}
for name, model in models.items():
    # 用标准化后的数据训练（KNN对量纲敏感，必须标准化）
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = round(acc, 4)
    print(f"{name} 测试集准确率：{acc:.4f}")

# 可视化：算法准确率对比柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(accuracy_results.keys(), accuracy_results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel("分类准确率", fontsize=12)
plt.title("不同机器学习算法的红酒品质分类准确率对比", fontsize=14)
plt.ylim(0.8, 1.0)  # 放大差异，更直观
# 在柱子上标注数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}', ha='center', va='bottom')
plt.savefig("算法准确率对比.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 3. 实验2：标准化对KNN的影响 =====================
# 未标准化的KNN
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))

# 标准化后的KNN
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))

print(f"\n未标准化KNN准确率：{acc_raw:.4f}")
print(f"标准化后KNN准确率：{acc_scaled:.4f}")

# 可视化：标准化前后对比
plt.figure(figsize=(8, 6))
plt.bar(["未标准化", "标准化后"], [acc_raw, acc_scaled], color=['#ff9999', '#66b3ff'])
plt.ylabel("分类准确率", fontsize=12)
plt.title("数据标准化对KNN算法准确率的影响", fontsize=14)
plt.ylim(0.6, 1.0)
for i, v in enumerate([acc_raw, acc_scaled]):
    plt.text(i, v+0.01, f'{v:.4f}', ha='center', fontsize=12)
plt.savefig("标准化效果对比.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 4. 实验3：K值对KNN准确率的影响（参数调优） =====================
k_range = range(1, 21)  # 测试K=1到20
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = accuracy_score(y_test, knn.predict(X_test_scaled))
    k_scores.append(score)

best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"\n最优K值：{best_k}，对应准确率：{best_score:.4f}")

# 可视化：K值准确率变化曲线
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o', color='#2ca02c', linewidth=2)
plt.xlabel("K值（邻居数量）", fontsize=12)
plt.ylabel("分类准确率", fontsize=12)
plt.title("KNN算法中K值与准确率的关系", fontsize=14)
plt.xticks(k_range)
plt.grid(alpha=0.3)
# 标注最高点
plt.scatter(best_k, best_score, color='red', s=100, label=f'最优K={best_k}')
plt.legend()
plt.savefig("K值调优曲线.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== 5. 补充：最优模型混淆矩阵 =====================
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_best = best_knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("最优KNN模型混淆矩阵", fontsize=14)
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names)
plt.yticks(tick_marks, target_names)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
# 标注数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.savefig("混淆矩阵.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n所有实验完成，图片已保存到项目文件夹！")