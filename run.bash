CUDA_VISIBLE_DEVICES=1,3 llamafactory-cli train \
    --stage sft \      #指定sft微调训练，可选rm，dpo等
    --do_train True \  #训练是do_train，预测是do_predict
    --model_name_or_path /data/LHRLLM/LLaMA-Factory-SPKL/model/glm-4-9b-chat-hf \  #模型目录，如果网络不行，可以配置本地目录，但今天的modelscope教程已经解决这个问题
    --finetuning_type lora \    #训练类型为lora，也可以进行full和freeze训练
    --quantization_bit 4 \      #量化精度，4bit，可选8bit和none不量化
    --template glm4 \      #模版，每个模型要选对应的模版，对应关系见上文
    --flash_attn auto \          #flash attention，闪光注意力机制，一种加速注意力计算的方法，后面会专门写一篇，baichuan2暂不支持，这里选auto，对于支持的模型可以选择fa2
    --dataset_dir data \        #数据目录
    --dataset jailbench \    #数据集，可以通过更改dataset_info.json文件配置自己的数据集
    --cutoff_len 1024 \         #截断长度
    --learning_rate 5e-05 \     #学习率，AdamW优化器的初始学习率
    --num_train_epochs 20.0 \   #训练轮数，需要执行的训练总轮数
    --max_samples 100000 \      #最大样本数，每个数据集的最大样本数
    --per_device_train_batch_size 1 \    #批处理大小，每个GPU处理的样本数量，推荐为1
    --gradient_accumulation_steps 1 \    #梯度累积，梯度累积的步数,推荐为1
    --lr_scheduler_type cosine \         #学习率调节器,可选line,constant等多种
    --max_grad_norm 1.0 \                #最大梯度范数,用于梯度裁剪的范数
    --logging_steps 100 \                #日志间隔，每两次日志输出间的更新步数
    --save_steps 5000 \                  #保存间隔，每两次断点保存间的更新步数。
    --warmup_steps 0.1 \                 #预热步数，学习率预热采用的步数。
    --optim adamw_torch \                #优化器，使用的优化器：adamw_torch、adamw_8bit 或 adafactor
    --packing False \                    
    --report_to none \
    --output_dir saves/GLM4-9B/lora/train_2024-12-9 \    #数据目录
    --fp16 True \                        #计算类型，可以fp16、bf16等
    --lora_rank 32 \                     #LoRA秩，LoRA矩阵的秩大小，越大精度越高，推荐32
    --lora_alpha 16 \                    #LoRA 缩放系数
    --lora_dropout 0 \
    --lora_target W_pack \               #模型对应的模块，具体对应关系见上文
    --val_size 0.1 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --per_device_eval_batch_size 1 \
    --load_best_model_at_end True \
    --plot_loss True