#!/usr/bin/env bash
# Input files:
host_filename="../../machinefiles/cogtwo"
data_filename="/home/yuntiand/downloads/imnet_feat.dat"
is_partitioned=false
data_format="binary"

# Ouput files:
output_dirname="output"
log_dirname="log"

# Sparse Coding parameters:
# Objective function parameters
m=21504
#n=1266734
n=126673
dictionary_size=10000
c=1.0
lambda=0.01
# Optimization parameters
num_epochs=500
minibatch_size=100
init_step_size_B=0.01
step_size_offset_B=0.0
step_size_pow_B=0.0
num_iter_S_per_minibatch=10
init_step_size_S=0.001
step_size_offset_S=0.0
step_size_pow_S=0.0
# Evaluation parameters
num_eval_minibatch=1
num_eval_samples=100

# System parameters:
num_worker_threads=4
table_staleness=100
maximum_running_time=0.0
load_cache=false
cache_dirname="N/A"

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=sparsecoding_main
prog_path=$app_dir/bin/$progname
data_file=$(readlink -f $data_filename)
host_file=$(readlink -f $host_filename)
log_path=${app_dir}/$log_dirname
output_path=${app_dir}/$output_dirname
cache_path=${app_dir}/$cache_dirname

# Mkdir
if [ ! -d "$output_path" ]; then
    mkdir -p $output_path
fi
if [ ! -d "$log_path" ]; then
    mkdir -p $log_path
fi

# Deal with booleans
if [ "$is_partitioned" = true ]; then
    flag_is_partitioned="is_partitioned"
else
    flag_is_partitioned="nois_partitioned"
fi 
if [ "$load_cache" = true ]; then
    flag_load_cache="load_cache"
else
    flag_load_cache="noload_cache"
fi 

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

  # input file name is assumed to be $data_file.$client_id if it is partitioned
  if [ "$is_partitioned" = true ]; then
      data_file_client="${data_file}.${client_id}"
  else
      data_file_client=$data_file
  fi 
  #cmd="GLOG_logtostderr=true \
  #cmd="setenv GLOG_logtostderr false; \
  #    setenv GLOG_log_dir $log_path; \
  #    setenv GLOG_v -1; \
  #    setenv GLOG_minloglevel 0; \
  #    setenv GLOG_vmodule ; \
  cmd="GLOG_logtostderr=false \
      GLOG_log_dir=$log_path \
      GLOG_v=-1
      GLOG_minloglevel=0 \
      GLOG_vmodule= \
      $prog_path \
      --hostfile $host_file \
      --data_file $data_file_client \
      --$flag_is_partitioned \
      --data_format $data_format \
      --output_path $output_path \
      --num_clients $num_unique_hosts \
      --num_worker_threads $num_worker_threads \
      --dictionary_size $dictionary_size \
      --m $m \
      --n $n \
      --c $c \
      --lambda $lambda \
      --num_epochs $num_epochs\
      --minibatch_size $minibatch_size \
      --num_iter_S_per_minibatch $num_iter_S_per_minibatch \
      --num_eval_minibatch $num_eval_minibatch \
      --num_eval_samples $num_eval_samples \
      --init_step_size_B $init_step_size_B \
      --step_size_offset_B $step_size_offset_B \
      --step_size_pow_B $step_size_pow_B \
      --init_step_size_S $init_step_size_S \
      --step_size_offset_S $step_size_offset_S \
      --step_size_pow_S $step_size_pow_S \
      --table_staleness $table_staleness \
      --maximum_running_time $maximum_running_time
      --$flag_load_cache
      --cache_path $cache_path
      --client_id $client_id"
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
