# colorful terminal output
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

yecho () {
  echo -e "${YELLOW}WARNING: $1${NC}"
}

gecho () {
  echo -e "${GREEN}$1${NC}"
}

recho () {
  echo -e "${RED}ERROR: $1${NC}"
}

if [ -f "$VITIS_PATH/settings64.sh" ];then
  source "$VITIS_PATH/settings64.sh"
  gecho "Found Vitis at $VITIS_PATH"
  if [ -f "$XILINX_XRT/setup.sh" ];then
    source "$XILINX_XRT/setup.sh"
    gecho "Found XRT at $XILINX_XRT"
  else
    recho "XRT not found on $XILINX_XRT. Check the installation in the Dockerfile"
    exit 1
  fi
else
  yecho "Unable to find $VITIS_PATH/settings64.sh"
  yecho "Please check that you correctly mounted the Xilinx installation path in the docker run command"
  exit 2
fi

exec "$@"
