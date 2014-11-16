#!/usr/bin/env bash
# Input files:
data_filename="/home/yuntiand/downloads/imnet_feat.dat"
host_filename="../../machinefiles/cogtwo"

# Sparse Coding parameters:
dictionary_size=10000
lambda=0.01
c=1.0
init_step_size=0.01
step_size_offset=50
step_size_pow=0.0
minibatch_size=100
num_eval_minibatch=1
num_iter_S_per_minibatch=10
init_step_size_S=0.01
step_size_offset_S=50
step_size_pow_S=0.0
# Execution parameters:
num_worker_threads=8
num_epochs=200

# System parameters:
staleness=100
table_staleness=$staleness

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=sparsecoding_main
prog_path=$app_dir/bin/${progname}
data_file=$(readlink -f $data_filename)
host_file=$(readlink -f $host_filename)
log_path=$app_dir/log_S_4
output_path=$app_dir/output_S

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
  #cmd="setenv GLOG_logtostderr false; \
  #    setenv GLOG_log_dir $log_path; \
  #    setenv GLOG_v -1; \
  #    setenv GLOG_minloglevel 0; \
  #    setenv GLOG_vmodule ""; \
  cmd="GLOG_logtostderr=false \
      GLOG_log_dir=$log_path \
      GLOG_v=-1
      GLOG_minloglevel=0 \
      GLOG_vmodule="" \
      $prog_path \
      --hostfile $host_file \
      --output_path $output_path \
      --num_clients $num_unique_hosts \
      --num_worker_threads $num_worker_threads \
      --dictionary_size $dictionary_size \
      --lambda $lambda \
      --init_step_size_B $init_step_size \
      --step_size_offset_B $step_size_offset \
      --step_size_pow_B $step_size_pow \
      --init_step_size_S $init_step_size_S \
      --step_size_offset_S $step_size_offset_S \
      --step_size_pow_S $step_size_pow_S \
      --num_iter_S_per_minibatch $num_iter_S_per_minibatch \
      --minibatch_size $minibatch_size \
      --num_eval_minibatch $num_eval_minibatch \
      --c $c \
      --data_file=${data_file} \
      --table_staleness $table_staleness \
      --num_epochs $num_epochs\
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
