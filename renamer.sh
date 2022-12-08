
echo "This is a shell script converting all the Chinese folder names into English ones.";

base_folder='Difficulty1_Hazy images stitching';
if [ -d "$base_folder" ]; then
    echo "name '$base_folder' is too long and contains [space], renaming to 'diff1/'";
    mv "$base_folder" "diff_1";
else
    if [ ! -d "diff_1" ]; then
        echo "Target folder not found. Exiting...";
        exit;
    fi
fi

cd diff_1/
mv "原始数据" "raw_data"
mv "实验结果" "results"

files=($(find . -name "无散射"))
for file in "${files[@]}"; do
    if [ -d $file ]; then
        echo "Chinese folder name found: '$file'"
        prefix=${file%/*}
        mv "$prefix/无散射/" "$prefix/no_scat/"
    fi
done

files=($(find . -name "散射"))
for file in "${files[@]}"; do
    echo "Chinese folder name found: '$file'"
    prefix=${file%/*}
    mv "$prefix/散射/" "$prefix/scat/"
done

echo "Process completed."
