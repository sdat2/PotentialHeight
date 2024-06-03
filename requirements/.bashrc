PATH=$PATH:/work/n02/n02/sdat2/bin
# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE='/mnt/lustre/a2fs-work2/work/n02/n02/sdat2/bin/micromamba';
export MAMBA_ROOT_PREFIX='/mnt/lustre/a2fs-work2/work/n02/n02/sdat2/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<
