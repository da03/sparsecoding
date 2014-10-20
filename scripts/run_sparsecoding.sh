#!/usr/bin/env bash
# Input files:
data_filename="data/data.mat"
host_filename="scripts/cogthree"

# Sparse Coding parameters:
dictionary_size=20000
lambda=1.0
c=1.0
init_step_size=0.01
step_size_offset=50
step_size_pow=0.0
mini_batch=1000
num_eval_minibatch=10
# Execution parameters:
num_worker_threads=4
num_iterations_per_thread=300

# System parameters:
staleness=0
table_staleness=$staleness

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=sparsecoding_main
prog_path=$app_dir/bin/${progname}
data_file=$(readlink -f $data_filename)
host_file=$(readlink -f $host_filename)

ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

# Parse hostfile
host_list=`cat $host_file | awk '{ print $2 }'`
unique_host_list=`cat $host_file | awk '{ print $2 }' | uniq`
num_unique_hosts=`cat $host_file | awk '{ print $2 }' | uniq | wc -l`

# Kill previous instances of this program
echo "Killing previous instances of '$progname' on servers, please wait..."
for ip in $unique_host_list; do
  ssh $ssh_options $ip \
    killall -q $progname
done
echo "All done!"

# Spawn program instances
client_id=0
for ip in $unique_host_list; do
  echo Running client $client_id on $ip

  #cmd="GLOG_logtostderr=true \
  cmd="GLOG_logtostderr=false \
      GLOG_v=-1 \
      GLOG_minloglevel=0 \
      GLOG_vmodule="" \
      $prog_path \
      --hostfile $host_file \
      --num_clients $num_unique_hosts \
      --num_worker_threads $num_worker_threads \
      --dictionary_size $dictionary_size \
      --lambda $lambda \
      --init_step_size $init_step_size \
      --step_size_offset $step_size_offset \
      --step_size_pow $step_size_pow \
      --mini_batch $mini_batch \
      --num_eval_minibatch $num_eval_minibatch \
      --c $c \
      --data_file=${data_file} \
      --table_staleness $table_staleness \
      --num_iterations_per_thread $num_iterations_per_thread \
      --client_id ${client_id}"

  ssh $ssh_options $ip $cmd &
  #eval $cmd  # Use this to run locally (on one machine).

  # Wait a few seconds for the name node (client 0) to set up
  if [ $client_id -eq 0 ]; then
    echo $cmd   # echo the cmd for just the first machine.
    echo "Waiting for name node to set up..."
    sleep 3
  fi
  client_id=$(( client_id+1 ))
done
