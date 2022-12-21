
echo "Brute force grid search for best param"

# parser.add_argument("--em_steps", type = int, default = 2, help = "E-M step for iterative matching re-estimation")
# parser.add_argument("--affinity_eps",   type = float, default = 30.0, help = "Sigma distance of allowed spatial inconsistency")
# parser.add_argument("--aff_thresh",     type = float, default = 0.5, help = "Threshold for spectral score replacement")     # baseline 5.0
# parser.add_argument("--epi_weight",     type = float, default = 0.5, help = "Weighting coeff for epipolar score")
# parser.add_argument("--fluc",           type = float, default = 0.5, help = "SDP fluctuation parameter for robust solution")
# parser.add_argument("--em_radius",      type = float, default = 6.0, help = "Point outside this radius after projection will not be considered")
# parser.add_argument("--score_thresh",   type = float, default = 0.4, help = "The matched features should have a similarity score above this threshold")
case_idx=1
if [ ! ''$1 = '' ]; then
    case_idx=$1
fi

em_steps=(1)
em_radius=(6)
score_thresh=(0.5)
affinity_eps=(20.0 30.0 40.0)
aff_thresh=(0.4 0.45 0.5 0.6)
epi_weight=(0. 0.5 1.0)
fluc=(0.05 0.2 0.4)

all_imgs=(1 2 4 5)

result_file="result_${case_idx}.txt"
echo "Grid search:" &> ${result_file}
all_cases=$((${#em_steps[@]}*${#affinity_eps[@]}*${#aff_thresh[@]}*${#epi_weight[@]}*${#fluc[@]}*${#em_radius[@]}*${#score_thresh[@]}))
cnt=0

for em_step in ${em_steps[@]};      do
for em_rads in ${em_radius[@]};     do
for score_t in ${score_thresh[@]};  do
for aff_eps in ${affinity_eps[@]};  do
for aff_thr in ${aff_thresh[@]};    do
for epi_wgt in ${epi_weight[@]};    do
for fluc_pa in ${fluc[@]};          do
    dir_path="output/case_${case_idx}"
    if [ ! -d ${dir_path} ]; then
        echo "Directory '${dir_path}' does not exist. Creating folder..."
        mkdir -p ${dir_path}
    fi

    for img_idx in ${all_imgs[@]}; do
        python3 ./spectral_method.py -m --case_idx ${case_idx} --img_idx ${img_idx} --only_diff                     \
                --em_steps ${em_step} --affinity_eps ${aff_eps}  --aff_thresh ${aff_thr} --epi_weight ${epi_wgt}    \
                --fluc ${fluc_pa} --em_radius ${em_rads} --score_thresh ${score_t}                                  \
                --baseline_hmat  
    done
    /usr/local/MATLAB/R2022a/bin/matlab -nodesktop -nodisplay -nosplash -r \
        "run('../diff_1/program/main_example.m');fprintf('${em_step}, ${aff_eps}, ${aff_thr}, ${epi_wgt}, ${fluc_pa}, ${em_rads}, ${score_t}');exit;" | tail -n +11 >> ${result_file}
    echo "" >> ${result_file}
    echo "Case ${cnt}/${all_cases} done."
    cnt=$(($cnt+1))
done
done
done
done
done
done
done