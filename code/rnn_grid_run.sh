set -e

gpu=$1 # 2070, 2080ti
output_dir=./rnn-epoch-latency-${gpu}

cmd_builder () {
  # $1: "normal" or "blelloch"
  # $2: seq_len
  # $3: batch_size
  # $4: output file name
  # $5: epochs
  # $6: extra flags

  cmd="python rnn.py \
    --save-dir=./bernoulli${2}_10/ \
    --rnn-type=cuDNN \
    --mode=$1 \
    --save-epoch-latency=$4 \
    --num-epochs=$5 \
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
  # $5: extra flags.

  cmd_builder $1 $2 $3 $4 $5 $6
  eval_cmd $cmd
}

mkdir -p $output_dir


for mode in normal blelloch
do
  run_experiment $mode 1000 16 \
    ${output_dir}/training-curve-${mode}-bernoulli1000_10-batch_size_16.csv \
    50 \
    "--save-loss-acc"
done

for batch_size in 2 4 8 16 32 64 128 256
do
  for mode in normal blelloch normal-nobp
  do
    run_experiment ${mode} 1000 $batch_size \
      ${output_dir}/${mode}-bernoulli1000_10-batch_size_${batch_size}.csv \
      10 \
      ""
  done
done

for seq_len in 10 30 100 300 3000 10000 30000
do
  for mode in normal blelloch normal-nobp
  do
    run_experiment $mode $seq_len 16 \
      ${output_dir}/${mode}-bernoulli${seq_len}_10-batch_size_16.csv \
      10 \
      ""
  done
done
