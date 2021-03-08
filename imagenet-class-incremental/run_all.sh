#Baseline
# LUCIR
CUDA_VISIBLE_DEVICES=4 python cbf_class_incremental_cosine_imagenet.py \
    --dataset imagenet --datadir data/imagenet/data/ --num_classes 1000 \
    --nb_cl_fg 500 --nb_cl 100 --nb_protos 20 \
    --resume --rs_ratio 0.0 --imprint_weights \
    --less_forget --lamda 10 --adapt_lamda \
    --random_seed 1993 \
    --mr_loss --dist 0.5 --K 2 --lw_mr 1 \
    --cb_finetune \
    --ckp_prefix cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_cosine_imagenet \
    2>&1 | tee ./logs/log_cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_cosine_imagenet_nb_cl_fg_500_nb_cl_100_nb_protos_20.txt

# evaluate
python cbf_eval_cumul_acc.py --nb_cl_fg 500 --nb_cl 100 --nb_protos 20 \
--run_id 0 \
--ckp_prefix cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_cosine_imagenet 

# add TDE evaluate
python cbf_eval_cumul_acc_TDE.py --nb_cl_fg 500 --nb_cl 100 --nb_protos 20 \
--run_id 0 \
--ckp_prefix cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_cosine_imagenet 


# Ours
# add DCE
CUDA_VISIBLE_DEVICES=2 python cbf_class_incremental_cosine_imagenet.py \
--dataset imagenet --datadir data/imagenet/data/ --num_classes 1000 \
--nb_cl_fg 500 --nb_cl 100 --nb_protos 20 \
--resume --rs_ratio 0.0 --imprint_weights \
--less_forget --lamda 10 --adapt_lamda \
--random_seed 1993 \
--mr_loss --dist 0.5 --K 2 --lw_mr 1 \
--cb_finetune \
--DCE --TDE --top_k 1 \
--ckp_prefixcbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_DCE_cosine_imagenet \
2>&1 | tee ./logs/log_cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_DCE_cosine_imagenet_nb_cl_fg_500_nb_cl_100_nb_protos_20.txt



