
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Urban AI Recommendation System - 直接启动Streamlit演示脚本
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'src'))


def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = ['pandas', 'numpy', 'sklearn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"警告：缺少以下依赖包：{', '.join(missing_packages)}")
        print("请运行：pip install -r requirements.txt")
        return False
    return True


def check_streamlit():
    """检查 Streamlit 是否已安装"""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def install_streamlit():
    """自动安装Streamlit"""
    print("正在自动安装 Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("Streamlit 安装成功！")
        return True
    except subprocess.CalledProcessError:
        print("Streamlit 安装失败，请手动运行: pip install streamlit")
        return False


def check_app_file():
    """检查app.py文件是否存在"""
    app_file = current_dir / 'src' / 'app.py'
    if not app_file.exists():
        print(f"错误：找不到Streamlit应用文件: {app_file}")
        print("请确保 src/app.py 文件存在")
        return False
    return True


def load_sample_data():
    """加载示例数据"""
    data_dir = current_dir / 'data'
    if not data_dir.exists():
        print("警告：data目录不存在")
        return None
        
    json_files = list(data_dir.glob('*.json'))

    if not json_files:
        print("警告：data目录中没有找到JSON文件")
        return None

    print(f"发现 {len(json_files)} 个数据文件：")
    for file in json_files:
        print(f"  - {file.name}")

    return len(json_files)


def main():
    """主函数 - 直接启动Streamlit"""
    print("=" * 60)
    print("Urban AI Recommendation System - Streamlit 启动器")
    print("=" * 60)

    # 1. 检查基础依赖
    print("\n1. 检查基础依赖...")
    if not check_dependencies():
        print("请先安装必要的依赖包，然后重新运行")
        return False

    # 2. 检查数据文件
    print("\n2. 检查数据文件...")
    data_count = load_sample_data()
    if data_count:
        print(f"✓ 数据文件检查完成")
    else:
        print("⚠ 未找到数据文件，但仍可运行应用")

    # 3. 检查app.py文件
    print("\n3. 检查Streamlit应用文件...")
    if not check_app_file():
        return False
    print("✓ 应用文件存在")

    # 4. 检查并安装Streamlit
    print("\n4. 检查Streamlit...")
    if not check_streamlit():
        print("Streamlit 未安装")
        install_choice = input("是否自动安装 Streamlit? (y/n): ").strip().lower()
        if install_choice == 'y':
            if not install_streamlit():
                return False
        else:
            print("请手动安装: pip install streamlit")
            return False
    else:
        print("✓ Streamlit 已安装")

    # 5. 启动Streamlit应用
    print("\n" + "=" * 60)
    print("🚀 正在启动 Streamlit 应用...")
    print("📱 应用将在浏览器中自动打开")
    print("🔗 如果没有自动打开，请手动访问显示的URL")
    print("⏹️  按 Ctrl+C 停止应用")
    print("=" * 60)

    try:
        # 启动streamlit应用
        app_path = current_dir / 'src' / 'app.py'
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            str(app_path),
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 应用已停止，感谢使用！")
    except FileNotFoundError:
        print("\n❌ 错误：无法找到Streamlit命令")
        print("请确保Streamlit已正确安装")
    except Exception as e:
        print(f"\n❌ 启动应用时发生错误：{e}")
        return False

    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ 启动失败，请检查上述错误信息")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n发生未预期的错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
