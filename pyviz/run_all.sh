
echo "This script generates all testing results..."

start_case=1
end_case=4

if [ ! ''$1 = '' ]; then
    start_case=$1
fi

if [ ! ''$2 = '' ]; then
    end_case=$2
fi

all_imgs=(1 2 4 5)

if [ ! -d output ]; then
    echo "Directory 'output/' does not exist. Creating folder..."
    mkdir output
fi

for ((case_idx=${start_case};case_idx<=${end_case};case_idx++)); do
    dir_path="output/case_${case_idx}"
    if [ ! -d ${dir_path} ]; then
        echo "Directory '${dir_path}' does not exist. Creating folder..."
        mkdir -p ${dir_path}
    fi
    for img_idx in ${all_imgs[@]}; do
        echo "Processing image ${img_idx} of case ${case_idx}..."
        python3 ./spectral_method.py --case_idx ${case_idx} --img_idx ${img_idx} -s --viz_kpt save --em_steps 2 --only_diff
    done
done

echo "Completed."