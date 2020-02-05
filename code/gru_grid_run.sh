set -e

gpu=$1 # 2070, 2080ti
epochs_train_curve=$2
epochs_bench=$3
output_dir=./gru-epoch-latency-${gpu}
lr=0.0003

cmd_builder () {
  # $1: "normal" or "blelloch"
  # $2: seq_len
  # $3: batch_size
  # $4: output file name
  # $5: number of epochs.
  # $6: extra flags.
  cmd="python gru.py \
    --save-dir=./IRMASmfcc_${2} \
    --rnn-type=cuDNN \
    --mode=$1 \
    --save-epoch-latency=$4 \
    --num-epochs=$5 \
    --learning-rate=${lr} \
    --train-batch-size=$3 $6"
}

eval_cmd () {
  echo "Running $*"
  eval $*
  sleep 60s
}

run_experiment () {
  # $1: "normal" or "blelloch"
  # $2: seq_len
  # $3: batch_size
  # $4: output file name
  # $5: number of epochs.
  # $6: extra flags

  cmd_builder $1 $2 $3 $4 $5 $6
  eval_cmd $cmd
}

mkdir -p $output_dir

for seq_len in s m l
do
  for mode in normal blelloch
  do
    run_experiment $mode $seq_len 16 \
      ${output_dir}/training-curve-${mode}-IRMASmfcc_${seq_len}-batch_size_16.csv \
      ${epochs_train_curve} \
      "--save-loss-acc"
  done
done

for batch_size in 16 32 64
do
  for seq_len in s m l
  do
    for mode in normal blelloch normal-nobp blelloch-nobp
    do
      run_experiment $mode $seq_len $batch_size \
        ${output_dir}/${mode}-IRMASmfcc_${seq_len}-batch_size_${batch_size}.csv \
        ${epochs_bench} \
        ""
    done
  done
done
