export DOWNLOAD_TO=/opt/tiger/fanzhiyun/data/ami
export PYANNOTE_DATABASE_CONFIG=${DOWNLOAD_TO}/database.yml
export TUTORIAL_DIR="$PWD/tutorials/data_preparation"
train_spk_path=./data/spk_info/train_spk
dev_spk_path=./data/spk_info/dev_spk_
test_spk_path=./data/spk_info/test_spk_

gpu=0
stage=1
down_rate=8

while [[ $# -gt 0 ]]
do
    case $1 in
    --gpu)
        shift
        gpu=$1
        shift
        ;;
    --stage)
        shift
        stage=$1
        shift
        ;;
    --down_rate)
        shift
        down_rate=$1
        shift
        ;;
    *)
        remained_args="${remained_args} $1"
        shift
        ;;
    esac
done &&

EXP_DIR=./exp/speaker_change_detection

echo "exp id: ${id}"
echo "down_rate: $down_rate"
echo "data path: $PYANNOTE_DATABASE_CONFIG"
echo "exp path: $EXP_DIR"
export TRN_DIR=${EXP_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train

#train
start_eps=1
end_eps=500
dev_every_eps=10
if [ $stage -le 1 ];then
    echo 'start to training'
    if [ $start_eps == 1 ]; then
        CUDA_VISIBLE_DEVICES=$gpu python3 ./pyannote/audio/applications/seqscd.py scd train --down_rate=$down_rate --subset=train  --to=$end_eps --parallel=4 ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset  --spk_path=$train_spk_path
    else
        CUDA_VISIBLE_DEVICES=$gpu python3 ./pyannote/audio/applications/seqscd.py scd train --down_rate=$down_rate --subset=train --from=$start_eps --to=$end_eps --parallel=4 ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset --spk_path=$train_spk_path
    fi
fi
# development 
if  [ $stage -le 2 ];then
    echo 'start to adjust threshold on the development set'
    CUDA_VISIBLE_DEVICES=$gpu python3 ./pyannote/audio/applications/seqscd.py scd validate --up_bound=1.0 --down_rate=$down_rate --oracle_vad=True --subset=development --from=1 --to=$end_eps --every=$dev_every_eps --batch 512 ${TRN_DIR} AMI.SpeakerDiarization.MixHeadset --spk_path=$dev_spk_path
fi
#test
if [ $stage -le 3 ];then
    echo 'start to test the best model'
    export VAL_DIR=${TRN_DIR}/validate_segmentation_fscore/AMI.SpeakerDiarization.MixHeadset.development/
    echo $VAL_DIR
    if [ ! -f ${VAL_DIR}/params.yml ];then
        echo 'You should run development stage first'
        exit 1
    fi
    best_threshold=`cat ${VAL_DIR}/params.yml | grep alpha`
    best_threshold=($best_threshold)
    best_threshold=${best_threshold[1]}
    ep=`cat ${VAL_DIR}/params.yml | grep epoch`
    ep=($ep)
    ep=${ep[1]}
    CUDA_VISIBLE_DEVICES=$gpu python3 ./pyannote/audio/applications/seqscd.py scd validate --up_bound=1.0 --down_rate=$down_rate --oracle_vad=True --subset=test --from=$ep --to=$ep --every=$dev_every_eps --batch 512 ${TRN_DIR} AMI.SpeakerDiarization.MixHeadset --spk_path=$test_spk_path --best_threshold=$best_threshold
fi
