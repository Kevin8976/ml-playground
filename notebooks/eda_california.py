from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 输出目录：脚本所在的 notebooks 目录
out_dir = Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)

# 加载数据
cal = fetch_california_housing(as_frame=True)
df = cal.frame.rename(columns=str.lower)
# 只选数值列（兼容各 pandas 版本）
num_cols = df.select_dtypes(include="number").columns

# 兼容不同版本的目标列命名：medhousevalue / medhouseval
candidates = [c for c in ("medhousevalue", "medhouseval") if c in df.columns]
if not candidates:
    raise KeyError(f"Target column not found. Columns = {list(df.columns)}")
target_col = candidates[0]

print(df[num_cols].describe().T)

# 相关性矩阵只用数值列
corr = df[num_cols].corr()


# 基本统计
print(df[num_cols].describe().T)
corr = df[num_cols].corr()


# 相关性热力图（只用 matplotlib，避免额外依赖）
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
plt.imshow(corr, interpolation="nearest")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
plt.tight_layout()
plt.savefig(out_dir / "corr_heatmap.png", dpi=150)
plt.close()
print("Saved ->", out_dir / "corr_heatmap.png")

# 基线线性回归
X = df[num_cols].drop(columns=[target_col])
y = df[target_col]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(Xtr, ytr)
pred = lr.predict(Xte)
mae = mean_absolute_error(yte, pred)
print("Baseline MAE:", mae)

# 目标分布直方图（用自动识别到的 target_col）
plt.figure()
df[target_col].hist(bins=30)
plt.title(f"Target distribution ({target_col})")
plt.tight_layout()
plt.savefig(out_dir / "target_hist.png", dpi=150)
plt.close()
print("Saved ->", out_dir / "target_hist.png")
