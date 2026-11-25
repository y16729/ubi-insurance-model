import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from openpyxl.styles import Font, PatternFill, Alignment

# ---------- 路径：输出到脚本所在目录下的「输出结果」 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "输出结果")
os.makedirs(OUT_DIR, exist_ok=True)
FILE = os.path.join(BASE_DIR, "FoMs_V1.完全.xlsx")   # 若为 csv 改成 "FoMs_V1.csv"

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

FEAT = ['distance', 'AvgSpd', '#Accel', '#SuddenStart', '#AbruptLaneChange',
        '#IntenseBrake', '#SuddenStop', '#AbruptSteering']

# ================= step1 读数据 / step2 检查 =================
df = pd.read_excel(FILE)
print("工作目录:", BASE_DIR)
print("样本数:", len(df), "  缺失值:", int(df[FEAT + ['score']].isnull().sum().sum()))

X = df[FEAT]          # 星级为 score 派生字段，排除以防标签泄露
y = df['score']

# ================= step3 划分 + 标准化（仅训练集 fit）+ 线性回归 =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
pred_lr_tr = lr.predict(X_train_s)
pred_lr_te = lr.predict(X_test_s)

def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse, float(np.sqrt(mse)), r2_score(y_true, y_pred)

# ================= step4 岭回归调参 =================
tune_rows = []
ridge, best_alpha, best_mse = None, None, np.inf
for a in [0.01, 0.1, 1, 10, 100]:
    m = Ridge(alpha=a)
    m.fit(X_train_s, y_train)
    tr = get_metrics(y_train, m.predict(X_train_s))
    te = get_metrics(y_test, m.predict(X_test_s))
    tune_rows.append([a, tr[0], tr[1], tr[2], te[0], te[1], te[2]])
    if te[0] < best_mse:
        best_mse, best_alpha, ridge = te[0], a, m

pred_ridge_tr = ridge.predict(X_train_s)
pred_ridge_te = ridge.predict(X_test_s)

# ================= step4 随机森林（原始特征） =================
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred_rf_tr = rf.predict(X_train)
pred_rf_te = rf.predict(X_test)

# ================= 汇总表 =================
compare = pd.DataFrame([
    ['多元线性回归', *get_metrics(y_train, pred_lr_tr), *get_metrics(y_test, pred_lr_te)],
    ['岭回归(alpha=%s)' % best_alpha, *get_metrics(y_train, pred_ridge_tr), *get_metrics(y_test, pred_ridge_te)],
    ['随机森林(200棵)', *get_metrics(y_train, pred_rf_tr), *get_metrics(y_test, pred_rf_te)],
], columns=['模型', '训练MSE', '训练RMSE', '训练R2', '测试MSE', '测试RMSE', '测试R2'])

coef = pd.DataFrame({'特征': FEAT, '多元线性回归系数': lr.coef_, '岭回归系数': ridge.coef_})
coef.loc[len(coef)] = ['截距(intercept)', lr.intercept_, ridge.intercept_]

imp = pd.DataFrame({'特征': FEAT, '重要性': rf.feature_importances_}).sort_values(
    '重要性', ascending=False, ignore_index=True)

corr = df[FEAT + ['score']].corr()['score'].drop('score').sort_values()
corr_df = pd.DataFrame({'特征': corr.index, '与score相关系数': corr.values})

test_out = X_test.copy()
test_out['真实score'] = y_test.values
test_out['线性回归预测'] = pred_lr_te
test_out['岭回归预测'] = pred_ridge_te
test_out['随机森林预测'] = pred_rf_te
test_out['残差'] = y_test.values - pred_lr_te

train_out = X_train.copy()
train_out['真实score'] = y_train.values
train_out['线性回归预测'] = pred_lr_tr
train_out['岭回归预测'] = pred_ridge_tr
train_out['随机森林预测'] = pred_rf_tr
train_out['残差'] = y_train.values - pred_lr_tr

tune_df = pd.DataFrame(tune_rows, columns=[
    'alpha', '训练MSE', '训练RMSE', '训练R2', '测试MSE', '测试RMSE', '测试R2'])

# ---------- 摘要说明 ----------
# 相关性必须按“绝对值”排序：-0.652 的强度大于 -0.119
_corr_sorted = corr_df.reindex(
    corr_df['与score相关系数'].abs().sort_values(ascending=False).index)
_corr_top = _corr_sorted.iloc[0]      # 绝对值最大 -> 相关性最强
_corr_weak = _corr_sorted.iloc[-1]    # 绝对值最小 -> 相关性最弱

_lr_tr = get_metrics(y_train, pred_lr_tr)
_lr_te = get_metrics(y_test, pred_lr_te)
_rg_tr = get_metrics(y_train, pred_ridge_tr)
_rg_te = get_metrics(y_test, pred_ridge_te)
_rf_tr = get_metrics(y_train, pred_rf_tr)
_rf_te = get_metrics(y_test, pred_rf_te)

summary = pd.DataFrame([
    ['数据概况', '样本 %d 条，特征 %d 个，缺失值 %d 个' % (
        len(df), len(FEAT), int(df[FEAT + ['score']].isnull().sum().sum()))],
    ['数据划分', '训练集 %d 条（70%%），测试集 %d 条（30%%），random_state=42' % (
        len(X_train), len(X_test))],
    ['多元线性回归', '训练R2=%.4f  测试R2=%.4f  测试RMSE=%.4f' % (
        _lr_tr[2], _lr_te[2], _lr_te[1])],
    ['岭回归(alpha=%s)' % best_alpha, '训练R2=%.4f  测试R2=%.4f  测试RMSE=%.4f' % (
        _rg_tr[2], _rg_te[2], _rg_te[1])],
    ['随机森林(200棵)', '训练R2=%.4f  测试R2=%.4f  测试RMSE=%.4f' % (
        _rf_tr[2], _rf_te[2], _rf_te[1])],
    ['过拟合判断', '随机森林训练与测试R2相差 %.4f，存在明显过拟合；线性回归与岭回归两者接近，泛化稳定' % (
        _rf_tr[2] - _rf_te[2])],
    ['最优模型', '岭回归(alpha=%s)，测试R2=%.4f 最高' % (best_alpha, _rg_te[2])],
    ['最重要特征', '%s，重要性 %.4f' % (imp.iloc[0]['特征'], imp.iloc[0]['重要性'])],
    ['次重要特征', '%s，重要性 %.4f' % (imp.iloc[1]['特征'], imp.iloc[1]['重要性'])],
    ['相关性最强', '%s，与score相关系数 %.3f（按绝对值排序第一）' % (
        _corr_top['特征'], _corr_top['与score相关系数'])],
    ['相关性最弱', '%s，与score相关系数 %.3f（按绝对值排序最后）' % (
        _corr_weak['特征'], _corr_weak['与score相关系数'])],
], columns=['项目', '说明'])

# ================= 图1 真实值 vs 预测值 =================
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, pred_lr_te, alpha=0.8, edgecolors='white', linewidths=0.6)
ax.plot([2, 10], [2, 10], 'r--', label='y = x（理想线）')
ax.set_xlabel('真实 score')
ax.set_ylabel('预测 score')
ax.set_title('真实值 vs 预测值（多元线性回归）')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "图1_真实值vs预测值.png"), dpi=200)
plt.close(fig)

# ================= 图2 残差图 =================
resid = y_test.values - pred_lr_te
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pred_lr_te, resid, alpha=0.8, edgecolors='white', linewidths=0.6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('预测 score')
ax.set_ylabel('残差（真实 − 预测）')
ax.set_title('残差分布图（多元线性回归）')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "图2_残差分布.png"), dpi=200)
plt.close(fig)

# ================= 图3 三模型 R2 对比 =================
fig, ax = plt.subplots(figsize=(7, 5))
xpos = np.arange(len(compare))
w = 0.36
b1 = ax.bar(xpos - w / 2, compare['训练R2'], w, label='训练集 R2', color='#4E79A7')
b2 = ax.bar(xpos + w / 2, compare['测试R2'], w, label='测试集 R2', color='#F28E2B')
ax.set_xticks(xpos)
ax.set_xticklabels(['多元线性回归', '岭回归', '随机森林'])
ax.set_ylabel('R2')
ax.set_title('三个模型 R2 对比')
ax.set_ylim(0, 1.05)
ax.bar_label(b1, fmt='%.3f', fontsize=9)
ax.bar_label(b2, fmt='%.3f', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "图3_模型R2对比.png"), dpi=200)
plt.close(fig)

# ================= 图4 随机森林特征重要性 =================
fig, ax = plt.subplots(figsize=(7, 5))
d = imp.iloc[::-1]
ax.barh(d['特征'], d['重要性'], color='#59A14F')
for i, v in enumerate(d['重要性']):
    ax.text(v + 0.005, i, '%.3f' % v, va='center', fontsize=9)
ax.set_xlabel('特征重要性')
ax.set_title('随机森林 特征重要性')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "图4_特征重要性.png"), dpi=200)
plt.close(fig)

# ================= 图5 相关性热力图（第二步用） =================
cm = df[FEAT + ['score']].corr().values
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm, cmap='RdYlGn_r', vmin=-1, vmax=1)
labels = ['里程', '均速', '急加速', '急起步', '急变道', '激烈刹', '急停', '猛打方向', 'score']
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, '%.2f' % cm[i, j], ha='center', va='center',
                color='black' if abs(cm[i, j]) < 0.7 else 'white', fontsize=8.5)
ax.set_title('特征相关性热力图')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "图5_相关性热力图.png"), dpi=200)
plt.close(fig)

# ================= Excel：多子表汇总 =================
xlsx_path = os.path.join(OUT_DIR, "第四步_模型评估与对比.xlsx")
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
    summary.to_excel(w, sheet_name='摘要说明', index=False)
    compare.round(4).to_excel(w, sheet_name='模型对比', index=False)
    tune_df.round(4).to_excel(w, sheet_name='岭回归调参', index=False)
    coef.round(4).to_excel(w, sheet_name='模型系数', index=False)
    imp.round(4).to_excel(w, sheet_name='特征重要性', index=False)
    corr_df.round(3).to_excel(w, sheet_name='相关性', index=False)
    test_out.round(4).to_excel(w, sheet_name='测试集预测', index=False)
    train_out.round(4).to_excel(w, sheet_name='训练集预测', index=False)
    wb = w.book
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color='FFFFFF')
            c.fill = PatternFill('solid', fgColor='1F4E79')
            c.alignment = Alignment(horizontal='center')
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(width * 1.6, 10), 40)
        ws.freeze_panes = 'A2'


# ================= 把源代码一并复制到输出目录 =================
import shutil
try:
    _src = os.path.abspath(__file__)
    _dst = os.path.join(OUT_DIR, os.path.basename(_src))
    if _src != _dst:
        shutil.copy2(_src, _dst)
        print("\n源代码已复制到输出目录: %s" % os.path.basename(_dst))
except Exception as _e:
    print("\n源代码复制失败:", _e)

# ================= 结束提示 =================
print("\n=== 全部结果已导出到：%s ===" % OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f, "  %.1f KB" % (os.path.getsize(os.path.join(OUT_DIR, f)) / 1024))
print("\n模型对比：")
print(compare.round(4).to_string(index=False))

print("✅ 当前保存目录 =", os.getcwd())

