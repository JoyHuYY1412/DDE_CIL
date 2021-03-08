# Baseline LUCIR
# train
CUDA_VISIBLE_DEVICES=1 python class_incremental_cosine_cifar100.py \
--nb_cl_fg 50 --nb_cl 10 --nb_protos 5 --resume \
--rs_ratio 0.0 --imprint_weights --less_forget --lamda 5 --adapt_lamda \
--random_seed 1993 --mr_loss --dist 0.5 --K 2 --lw_mr 1 \
--TDE \
--ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100 \
2>&1 | tee ./logs/log_seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt

# evaluate
python eval_cumul_acc.py --nb_cl_fg 50 --nb_cl 10 --nb_protos 5 \
--run_id 0 \
--ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100

# add TDE evaluate
python eval_cumul_acc_TDE.py --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
--run_id 0 \
--ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100

# Ours
# add DCE
CUDA_VISIBLE_DEVICES=2 python class_incremental_cosine_cifar100.py \
 --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 --resume \
--rs_ratio 0.0 --imprint_weights --less_forget --lamda 5 --adapt_lamda \
--random_seed 1993 --mr_loss --dist 0.5 --K 2 --lw_mr 1 \
--DCE --TDE --top_k 10 \
--ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_DCE_cosine_cifar100 \
2>&1 | tee ./logs/log_seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_DCE_cosine_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt

# DCE + TDE evaluate
python eval_cumul_acc_TDE.py --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
--DCE --top_k 10 \
--run_id 0 \
--ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_DCE_cosine_cifar100
