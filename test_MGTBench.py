import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import csv
import sys
import os
# from binoculars.detector_gpt2 import Binoculars
from binoculars.detector import Binoculars



# def parse_args():
#     parser = argparse.ArgumentParser(description="Test Binoculars model on text datasets.")
#     parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/cyg/ai_text_detection/Datasets/MGTBench_datasets/Essay_LLMs.csv', help='Path to the dataset CSV file')
#     parser.add_argument('--results_path', type=str, default='/root/autodl-tmp/cyg/ai_text_detection/Binoculars/test_results/test_results.csv', help='Path to save the CSV results')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model predictions')
#     parser.add_argument('--min_tokens', type=int, default=64, help='Minimum number of tokens required to run predictions')
#     return parser.parse_args()


# 假设模型和数据集加载的函数
# 内有硬编码
def load_mgtbench_one(data_path, llm_generated, test_mode):
    # 示例数据加载
    data = pd.read_csv(data_path)
    # 增加两列 'label_human' 和 'label_AI' 并填充值

    data['label_human'] = 0
    data['label_AI'] = 1
    
    if test_mode == 'many':
        combined_texts = pd.concat([data['human'], data[llm_generated]], ignore_index=True)
        combined_label = pd.concat([data['label_human'], data['label_AI']], ignore_index=True)
    else:
        combined_texts = pd.concat([data['human'][1:33], data[llm_generated][1:33]], ignore_index=True)
        combined_label = pd.concat([data['label_human'][1:33], data['label_AI'][1:33]], ignore_index=True)
    
    return combined_texts, combined_label

# def load_mage():
#     # 示例数据加载
#     data = pd.read_csv('path/to/mage.csv')
#     return data['text'], data['label']

def load_model_and_tokenizer():
    # 示例模型加载
    # model = BinocularsDetector('path/to/model')
    model = Binoculars()
    tokenizer = model.tokenizer
    return model,tokenizer


def model_predict_batch(model, tokenizer, batch_texts, MINIMUM_TOKENS):
    batch_predictions = []
    batch_Prediction_failed_conditions = []
    for text in batch_texts:
        # 确保文本是字符串类型
        if not isinstance(text, str):
            batch_predictions.append("Error: Text must be a string")
            continue

        #处理不能被分词的异常
        # try:
        #     encoded_text = tokenizer(text)
        # except ValueError as e:
        #     batch_predictions.append(f"Error: {str(e)}")
        #     continue

        if len(tokenizer(text).input_ids) < MINIMUM_TOKENS:
            batch_predictions.append("Error: Text too short")  
        else:
            prediction = model.predict(text)
            if prediction not in ["Most likely AI-generated", "Most likely human-generated"]:
                batch_predictions.append("Error: Prediction failed")
                batch_Prediction_failed_conditions.append(prediction)
            else:
                batch_predictions.append(prediction)
                
    return batch_predictions, batch_Prediction_failed_conditions


def evaluate_model(model, tokenizer, texts, labels, MINIMUM_TOKENS, batch_size):
    assert len(texts) == len(labels), "The length of texts and labels must be equal."

    start_time = time.time()
    predictions = []
    Prediction_failed_conditions = []

    total_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总批次数
    print(f"Total batches: {total_batches}")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_predictions,batch_Prediction_failed_conditions = model_predict_batch(model, tokenizer, batch_texts, MINIMUM_TOKENS)
        predictions.extend(batch_predictions) #将批预测结果添加到预测结果末尾
        Prediction_failed_conditions.extend(batch_Prediction_failed_conditions)

        # 打印进度
        processed_batches = i // batch_size + 1
        #if processed_batches % 10 == 0 or processed_batches == total_batches:
        print(f"Processed {processed_batches}/{total_batches} batches.")

    #移除掉跳过的预测和相应的标签
    valid_predictions = []
    valid_labels = []
    for pred, label in zip(predictions, labels):
        if "Error" not in pred:
            valid_predictions.append(1 if pred == "Most likely AI-generated" else 0)
            valid_labels.append(label)

    end_time = time.time()

    print(f'预测失败的情况：{Prediction_failed_conditions}')

    accuracy = accuracy_score(valid_labels, valid_predictions)
    precision = precision_score(valid_labels, valid_predictions)
    recall = recall_score(valid_labels, valid_predictions)
    f1 = f1_score(valid_labels, valid_predictions)
    

    total_time = end_time - start_time
    avg_time = total_time / len(valid_predictions) if valid_predictions else 0
    return accuracy, precision, recall, f1, total_time, avg_time


def save_results_to_csv(results, file_path):
     # 检查文件是否存在以确定是否需要写入表头
    file_exists = os.path.isfile(file_path)
    initial_count = 0
    
    # 如果文件存在，计算已有的测试次数（不包括表头）
    if file_exists:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            initial_count = sum(1 for row in reader) - 1 # 减去表头


    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 表头
        headers = ['Test Index', 'Detect_Model', 'Dataset','LLM_Generated', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Total Prediction Time (s)', 'Avg Prediction Time (s)']

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(headers)

        # 写入数据，包括索引
        # 写入数据，包括索引
        initial_count += 1  # 更新索引
        index_and_results = [initial_count] + results  # 将索引和结果合并成一行
        writer.writerow(index_and_results)  # 写入整行数据



# LLM_Generated = 'ChatGPT'  
LLM_Generated_list = ['ChatGPT-turbo', 'Claude', 'ChatGLM', 'Dolly','ChatGPT', 'GPT4All', 'StableLM']
save_file_path = '/root/autodl-tmp/cyg/ai_text_detection/Binoculars/test_results/test_results.csv'
# MGTBench_domain = 'MGTBench_Essay' 
MGTBench_domain_list =  ['MGTBench_Essay', 'MGTBench_Reuters', 'MGTBench_WP']

data_path_list = ['/root/autodl-tmp/cyg/ai_text_detection/Datasets/MGTBench_datasets/Essay_LLMs.csv',
'/root/autodl-tmp/cyg/ai_text_detection/Datasets/MGTBench_datasets/Reuters_LLMs.csv',
'/root/autodl-tmp/cyg/ai_text_detection/Datasets/MGTBench_datasets/WP_LLMs.csv'
]
# data_path = '/root/autodl-tmp/cyg/ai_text_detection/Datasets/MGTBench_datasets/Essay_LLMs.csv'
test_mode = 'many'  #需要修改模型，little使用gpt2，many测试所有数据

domain_dict = dict(zip(MGTBench_domain_list, data_path_list))


# 加载模型
binoculars_model, binoculars_tokenizer  = load_model_and_tokenizer()


for MGTBench_domain in MGTBench_domain_list:
    data_path = domain_dict.get(MGTBench_domain, "domain is not found")

    for LLM_Generated in LLM_Generated_list:

        # 加载数据集
        try:
            mgtbench_one_texts, mgtbench_one_labels = load_mgtbench_one(
                data_path = data_path,
                llm_generated = LLM_Generated,
                test_mode = test_mode
                )
        except FileNotFoundError:
            print("指定的文件不存在，请检查文件路径")
            sys.exit(1)


        # 在MGTBench数据集上测试Binoculars
        print(f"Testing Binoculars on {MGTBench_domain}...")
        accuracy, precision, recall, f1, total_time, avg_time = evaluate_model(
            binoculars_model, 
            binoculars_tokenizer, 
            mgtbench_one_texts, 
            mgtbench_one_labels, 
            MINIMUM_TOKENS=1, #1为不限制文本长度，原来为64
            batch_size=32
            )

        # 存储所有测试结果的列表
        results = []
        results.extend(['Binoculars', MGTBench_domain, LLM_Generated, accuracy, precision, recall, f1, total_time, avg_time])
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, total Prediction Time (s): {total_time:.2f}, Avg Prediction Time (s): {avg_time:.2f}")


        # # 在MAGE数据集上测试Binoculars
        # print("Testing Binoculars on MAGE...")
        # accuracy, precision, recall, f1, total_time, avg_time = evaluate_model(binoculars_model, mage_texts, mage_labels)
        # results.append(['Binoculars', 'MAGE', accuracy, precision, recall, f1, total_time, avg_time])
        # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, total Prediction Time (s): {total_time:.2f}, Avg Prediction Time (s): {avg_time:.2f}")



        # 保存测试结果到CSV文件
        save_results_to_csv(results, file_path=save_file_path)
        print("Results saved to test_results.csv")
        
        print(pd.read_csv(save_file_path))

