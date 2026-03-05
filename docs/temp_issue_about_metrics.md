# 当前训练与评估指标存在的问题
## 涉及文件
使用配置文件configs/phase1_actions_curriculum_stage1_accel.yaml训练得到的数据，存放于runs/stage1_accel文件夹中。 
评估数据生成：
1. python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir runs/stage1_accel --episodes 200  --hybrid_bw_sat queue_aware
2. python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir runs/stage1_accel --episodes 200  --baseline queue_aware  
相关的描述说明文档：
docs/metrics_guide.md

## 现有问题
一些指标有对应数据，但说明文档中未写明，需要补充；一些指标已经不监测但说明文档尚未修改；一些指标对应数据一直是0，是否正常；是否还需添加其他指标，如KL散度、优势函数的均值/标准差等。
1. 文档中未写明的：actor_lr arrival_rate_eff arrival_sum centroid_dist_mean critic_lr outflow_sum imitation_coef imitation_loss queue_total (为何数量级比gu_queue_max小，为何gu_queue_max与gu_queue_mean相差4个数量级，是否为统计错误)queue_total_active drop_sum r_assoc_ratio r_bw_align r_centroid r_dist r_dist_delta r_drop_ratio r_energy r_fail_penalty r_queue_delta r_queue_pen r_queue_topk r_sat_score r_service_ratio r_term_assoc r_term_bw_align r_term_centroid r_term_dist r_term_dist_delta r_term_drop r_term_energy r_term_q_delta r_term_queue r_term_sat_score r_term_service r_term_topk reward_raw service_norm 
2. 一直为空或不变的，是设置原因还是计算出错？drop_norm gu_drop_sum imitation_loss r_assoc_ratio(一直为1) r_dist r_dist_delta r_drop_ratio r_queue_topk  r_service_ratio r_term_assoc r_term_bw_align r_term_dist r_term_dist_delta r_term_drop r_term_energy r_term_sat_score r_term_service r_term_topk uav_drop_sum  uav_queue_max uav_queue_mean

1. 不要并行读取 
2. 执行代码前先激活虚拟环境 
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```