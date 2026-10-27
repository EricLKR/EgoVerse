#!/usr/bin/env bash
# 在每个标定终端中 source 本文件。
if [[ -n "${ZSH_VERSION:-}" ]]; then
  eval '_calib_env_file=${(%):-%x}'
  _calib_shell=zsh
else
  _calib_env_file="${BASH_SOURCE[0]}"
  _calib_shell=bash
fi
_calib_root="$(cd -- "$(dirname -- "$_calib_env_file")/.." && pwd)"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo '缺少 ROS 2 Jazzy，请先运行 bash scripts/install_system.sh。' >&2
  return 1
fi
if [[ ! -f "${_calib_root}/ros2_ws/.venv/bin/activate" ]]; then
  echo '缺少 ros2_ws/.venv，请按 README 创建 Python 环境。' >&2
  return 1
fi
source "/opt/ros/jazzy/setup.${_calib_shell}" || return 1
export PYTHONNOUSERSITE=1
# 给本机 RGB 图像足够的 DDS 共享内存；保留用户已指定的传输配置。
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${_calib_root}/config/fastdds.xml}"
export EASY_HANDEYE2_SAMPLES_DIRECTORY="${_calib_root}/calibration/samples"
export EASY_HANDEYE2_CALIBRATIONS_DIRECTORY="${_calib_root}/calibration/native"
source "${_calib_root}/ros2_ws/.venv/bin/activate"
# ROS 的系统 shebang 脚本（包括 rqt）也要使用相同的依赖版本。
_calib_site="$("${_calib_root}/ros2_ws/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="${_calib_site}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${_calib_root}/ros2_ws/install/setup.bash" ]]; then
  source "${_calib_root}/ros2_ws/install/setup.${_calib_shell}" || return 1
fi
unset _calib_root _calib_site _calib_shell _calib_env_file
