case_idx=1
img_idx=1

if [ ! ''$1 = '' ]; then
    case_idx=$1
fi

if [ ! ''$2 = '' ]; then
    img_idx=$2
fi

dir_path="output/case_${case_idx}"
if [ ! -d ${dir_path} ]; then
    echo "Directory '${dir_path}' does not exist. Creating folder..."
    mkdir -p ${dir_path}
fi
echo "Processing image ${img_idx} of case ${case_idx}..."
python3 ./spectral_method.py --case_idx ${case_idx} --img_idx ${img_idx} -s -m --viz_kpt save --only_diff --viz ransac \
    --config ./configs/case${case_idx}.txt  --baseline_hmat
