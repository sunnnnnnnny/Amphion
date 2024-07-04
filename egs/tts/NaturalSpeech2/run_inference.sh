# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))
echo ${work_dir}
echo ${exp_dir}
export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config.json"
exp_name="ns2_libritts"
ref_audio="$work_dir/egs/tts/NaturalSpeech2/prompt_example/LJ050-0278.wav"
checkpoint_path="/Users/zhangsan/workspace/model_hg_temp/amphion_ns2/epoch-0089_step-0512912_loss-6.367693"
output_dir="$exp_dir/output"
mode="single"

export CUDA_VISIBLE_DEVICES="0"

######## Parse Command Line Arguments ###########
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --text)
    text="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

echo ${exp_config}
echo ${mode}
echo ${checkpoint_path}
echo ${ref_audio}
echo ${output_dir}

exit 1;
######## Train Model ###########
python "${work_dir}"/bins/tts/inference.py \
    --config=$exp_config \
    --text="hello world" \
    --mode=$mode \
    --checkpoint_path=$checkpoint_path \
    --ref_audio=$ref_audio \
    --output_dir=$output_dir