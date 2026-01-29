Steo 1:要使用 Anaconda 快速设置您的环境：conda env create -f environment.yml
                                     conda activate llava

Step 2:参考https://github.com/YukeHu/vlm_mia中数据集模型的下载和Conversation Generation生成对话
Step 3:在代码中的data = load_data('放入你的对话文件')，model_name='放入你的encoder路径'
data_ablation.py用于验证 “最优 z（几何中位数）
如data_consine_diff.py这类_diff结尾的文件用于对比不同阈值下的结果