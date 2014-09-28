#!/usr/bin/env bash

# Input files:
data_filename="data/mat.data"
host_filename="scripts/localserver"

# Sparse Coding parameters:
dictionary_size=0
lambda=1
c=1.0

# Execution parameters:
num_worker_threads=4
num_iterations_per_thread=100

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

output_dir=$app_dir/output
output_dir="${output_dir}/lda.S${staleness}"
output_dir="${output_dir}.M${num_unique_hosts}"
output_dir="${output_dir}.T${num_worker_threads}"
rm -rf ${output_dir}
mkdir -p ${output_dir}

output_file_prefix=${output_dir}/sparsecoding_out  # Prefix for program output files.

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

  cmd="GLOG_logtostderr=true \
      GLOG_v=-1 \
      GLOG_minloglevel=0 \
      GLOG_vmodule="" \
      $prog_path \
      --hostfile $host_file \
      --num_clients $num_unique_hosts \
      --num_worker_threads $num_worker_threads \
      --dictionary_size $dictionary_size \
      --lambda $lambda \
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
