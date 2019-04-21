#!/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo ${machine}

bash_rc=.bash_rc

cur_dir=`pwd`
LINUX_CPU_URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz
OS_X_CPU_URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.12.0.tar.gz
LINUX_GPU_URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz  

echo "download tensorflow-1.2 c api release CPU"
if [ "$machine" = "Linux" ] && [ ! -d "$cur_dir/include" ] ; then
    echo "download linux version"
    wget $LINUX_GPU_URL 
    file_name=`ls *.tar.gz`
    tar xzvf $file_name
    rm $file_name

elif [ "$machine" = "Mac" ] && [ ! -d "$cur_dir/include" ] ; then
    echo "download mac version"
    curl -O $OS_X_CPU_URL
    file_name=`ls *.tar.gz`
    tar xzvf $file_name
    rm $file_name
    bash_rc=.bash_profile
    if [[ "$DYLD_LIBRARY_PATH" != *"${cur_dir}"* ]]; then 
        echo 'export lib/ to DYLD_LIBRARY_PATH'
        echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$cur_dir/lib" >>~/${bash_rc}
    fi
fi 

echo $cur_dir
if [[ "$LD_LIBRARY_PATH" != *"${cur_dir}"* ]]; then 
   echo 'export lib/ to LD_LIBRARY_PATH'
   echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$cur_dir/lib" >>~/${bash_rc}
fi

if [[ "$LIBRARY_PATH" != *"${cur_dir}"* ]]; then
   echo 'export lib/ to LIBRARY_PATH'
   export LIBRARY_PATH=$LIBRARY_PATH:$cur_dir/lib
   echo "export LIBRARY_PATH=\$LIBRARY_PATH:$cur_dir/lib" >>~/${bash_rc}
fi

if [[ $CPATH != *"$cur_dir"* ]]; then
   echo 'export include/ to CPATH'
   echo "export CPATH=\$CPATH:$cur_dir/include" >>~/${bash_rc}
fi

echo "please restart your shell"
#export C_INCLUDE_PATH=$C_INCLUDE_PATH:$cur_dir/include
#export CPLUS_INCUDE_PATH=$CPLUS_INCLUDE_PATH:$cur_dir/include
