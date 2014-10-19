#!/usr/bin/env bash
# Input files:
data_filename="data/data.mat"
host_filename="scripts/cogthree"

# Sparse Coding parameters:
dictionary_size=0
lambda=1.0
c=1.0
init_step_size=0.00005
step_size_offset=50
step_size_pow=0.0
mini_batch=10
num_eval_minibatch=100
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

