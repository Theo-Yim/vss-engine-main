################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

# additional components the user can self install
apt-get update
apt-get install -y gstreamer1.0-libav
# ubuntu 22.04
apt-get install --reinstall -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly  libswresample-dev libavutil-dev libavutil56 \
    libavcodec-dev libavcodec58 libavformat-dev libavformat58 libavfilter7 \
    libde265-dev libde265-0 libx265-199 libx264-163 libvpx7 libmpeg2encpp-2.1-0 \
    libmpeg2-4 libmpg123-0 libswresample3 libjack0 libzvbi0 libxvidcore4 libslang2 \
    libflac8
ldconfig

echo "Deleting GStreamer cache"
rm -rf ~/.cache/gstreamer-1.0/