tmp=$(pwd)
mkdir $tmp/matplotlib
echo 'export MPLCONFIGDIR='$tmp'/matplotlib' >> ~/.bashrc
source ~/.bashrc