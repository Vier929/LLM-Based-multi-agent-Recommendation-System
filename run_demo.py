#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Urban AI Recommendation System - 演示启动脚本（修复版）
"""

import os
import sys
import json
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


def load_sample_data():
    """加载示例数据"""
    data_dir = current_dir / 'data'
    json_files = list(data_dir.glob('*.json'))

    if not json_files:
        print("警告：data目录中没有找到JSON文件")
        return None

    print(f"找到 {len(json_files)} 个数据文件")
    for file in json_files:
        print(f"  - {file.name}")

    # 加载第一个JSON文件作为示例
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data, json_files[0].name


def run_demo():
    """运行演示"""
    print("=" * 50)
    print("Urban AI Recommendation System - 演示程序")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        return

    # 加载数据
    print("\n正在加载数据...")
    result = load_sample_data()

    if result:
        data, filename = result
        print(f"\n数据加载成功！从文件: {filename}")
        print(f"数据概览：")

        # 处理不同类型的数据
        if isinstance(data, list):
            print(f"  - 类型: 列表 (List)")
            print(f"  - 记录数: {len(data)}")
            if data:
                print(f"  - 第一条记录的键: {list(data[0].keys()) if isinstance(data[0], dict) else '非字典类型'}")
        elif isinstance(data, dict):
            print(f"  - 类型: 字典 (Dict)")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} 条记录")
                else:
                    print(f"  - {key}: {type(value).__name__}")
        else:
            print(f"  - 类型: {type(data).__name__}")

    # 检查是否是 Streamlit 应用
    app_file = current_dir / 'src' / 'app.py'
    if app_file.exists():
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'streamlit' in content:
                print("\n检测到 app.py 是 Streamlit 应用！")
                print("请使用以下命令运行：")
                print(f"  streamlit run {app_file}")
                print("\n如果没有安装 streamlit，请运行：")
                print("  pip install streamlit")


def check_streamlit():
    """检查 Streamlit 是否已安装"""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n" + "=" * 30)
        print("请选择操作：")
        print("1. 运行推荐系统演示")
        print("2. 查看数据文件")
        print("3. 检查环境配置")
        print("4. 运行 Streamlit 应用")
        print("5. 退出")
        print("=" * 30)

        choice = input("请输入选项 (1-5): ").strip()

        if choice == '1':
            run_demo()
        elif choice == '2':
            data_dir = current_dir / 'data'
            files = list(data_dir.glob('*'))
            if files:
                print("\nData目录内容：")
                for f in files:
                    print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

                # 显示每个JSON文件的内容预览
                json_files = [f for f in files if f.suffix == '.json']
                if json_files:
                    preview = input("\n查看文件内容预览？(y/n): ").strip().lower()
                    if preview == 'y':
                        for json_file in json_files[:3]:  # 只显示前3个文件
                            print(f"\n--- {json_file.name} ---")
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        print(f"列表，包含 {len(data)} 个元素")
                                        if data and isinstance(data[0], dict):
                                            print(
                                                f"第一个元素: {json.dumps(data[0], ensure_ascii=False, indent=2)[:200]}...")
                                    elif isinstance(data, dict):
                                        print(f"字典，包含键: {list(data.keys())}")
                            except Exception as e:
                                print(f"读取错误: {e}")
            else:
                print("\nData目录为空")
        elif choice == '3':
            print("\nPython版本：", sys.version)
            print("当前路径：", os.getcwd())
            print("项目路径：", current_dir)
            check_dependencies()
            print(f"\nStreamlit 已安装: {'是' if check_streamlit() else '否'}")
        elif choice == '4':
            if check_streamlit():
                print("\n正在启动 Streamlit 应用...")
                print("提示：在浏览器中打开显示的 URL")
                import subprocess
                subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])
            else:
                print("\nStreamlit 未安装！")
                print("请运行: pip install streamlit")
        elif choice == '5':
            print("\n感谢使用！再见！")
            break
        else:
            print("\n无效选项，请重试")


if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n发生未预期的错误：{e}")
        import traceback

        traceback.print_exc()