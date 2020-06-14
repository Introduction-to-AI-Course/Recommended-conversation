# Recommended-conversation 

## 代码结构
```
Conversational-Recommendation-BASELINE
├── requirements.txt                  # 第三方依赖
├── README.md                         # 本文档
└── conversational-recommendation     # 源码
├── generative_model                  # 生成模型
│   ├── data                          # 数据
│   ├── models                        # 默认模型保存路径
│   ├── network.py                    # 模型配置、训练和测试
│   ├── output                        # 默认输出路径
│   ├── run_test.sh                   # 测试脚本
│   ├── run_train.sh                  # 训练脚本
│   ├── source                        # 模型的实现
│   └── tools                         # 工具
├── goal_planning                     # 对话目标规划
    ├── logs                          # 保存的log
    ├── data_generater                # 生成训练所需数据
    ├── process_data                  # 处理后的数据     
    ├── model                         # 模型
    ├── model_state                   # 默认模型保存路径
    ├── train_data                    # 转换为模型所需数据
    └── origin_data                   # 原始数据

```

## 注意事项
1. File goal_planning/origin_data/train.txt is 262.48 MB; this exceeds GitHub's file size limit of 100.00 MB 所以goal_planning/origin_data/文件夹下的数据要靠大家自行下载了



## 参考资料
[1] Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. 2018. Towards deep conversational recommendations. In NIPS
[2] Qibin Chen, Junyang Lin, Yichang Zhang, Ming Ding, Yukuo Cen, Hongxia Yang, and Jie Tang. 2019. Towards knowledge-based recommender dialog system. In EMNLP
[3] [AI Studio 基线系统示例AI Studio baseline model](https://aistudio.baidu.com/aistudio/projectdetail/360479)
