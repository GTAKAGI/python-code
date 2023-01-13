import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# 乱数の発生
generator = np.random.default_rng(100)
rnd = generator.normal(size=(10,10))
 
# ヒートマップの作成
sns.heatmap(rnd)
plt.show()