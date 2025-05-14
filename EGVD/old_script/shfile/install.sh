#!/bin/bash

# # 无限循环，直到安装成功
# while true; do
#     echo "尝试安装依赖项..."
    
#     # 尝试安装
#     pip install -r req.txt
    
#     # 检查上一次命令的退出码
#     if [ $? -eq 0 ]; then
#         echo "依赖项安装成功！"
#         break
#     else
#         echo "安装失败，重新尝试..."
#         sleep 2  # 等待2秒后再次尝试
#     fi
# done

# 无限循环，直到安装成功
while true; do
    echo "尝试安装依赖项..."
    
    # 尝试安装
    bash keyframe_interpolation.sh
    
    # 检查上一次命令的退出码
    if [ $? -eq 0 ]; then
        echo "依赖项安装成功！"
        break
    else
        echo "安装失败，重新尝试..."
        sleep 2  # 等待2秒后再次尝试
    fi
done