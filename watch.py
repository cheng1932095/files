import pandas as pd

# 设置pandas显示所有列
pd.set_option('display.max_columns', None)
data = pd.read_csv('/root/autodl-tmp/cyg/ai_text_detection/Datasets/MGTBench_datasets/WP_LLMs.csv')
# 增加两列 'label_human' 和 'label_AI' 并填充值
data['label_human'] = 0
data['label_AI'] = 1
print(data.columns)
print(data.head())
print(data.info)


combined_label = pd.concat([data['label_human'], data['label_AI']], ignore_index=True)
print(combined_label)



labels = data['label_AI']
labels = labels.tolist()
print(labels)
print(len(labels))
labels.remove(1)
print(labels)
print(len(labels))