# Urban AI Demo Package Setup Script
# 运行此脚本来完成项目设置

Write-Host "Urban AI Recommendation System - 项目设置" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 设置项目路径
$projectPath = "C:\Users\luvyf\Desktop\UrbanAI_Demo_Package"

# 检查项目目录是否存在
if (-not (Test-Path $projectPath)) {
    Write-Host "错误：项目目录不存在！" -ForegroundColor Red
    exit 1
}

Set-Location $projectPath
Write-Host "`n当前目录: $projectPath" -ForegroundColor Yellow

# 创建目录结构
Write-Host "`n创建项目目录结构..." -ForegroundColor Cyan
$directories = @("src", "data", "docs", "tests", "logs", "config")

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  ✓ 创建目录: $dir" -ForegroundColor Green
    } else {
        Write-Host "  - 目录已存在: $dir" -ForegroundColor Gray
    }
}

# 移动文件到正确位置
Write-Host "`n整理项目文件..." -ForegroundColor Cyan

# 移动Python文件到src目录
if (Test-Path "recommendation+system.py") {
    Move-Item -Path "recommendation+system.py" -Destination "src\" -Force
    Write-Host "  ✓ 移动 recommendation+system.py 到 src/" -ForegroundColor Green
}

# 移动JSON文件到data目录
$jsonFiles = Get-ChildItem -Path . -Filter "*.json" -File
if ($jsonFiles) {
    foreach ($file in $jsonFiles) {
        Move-Item -Path $file.FullName -Destination "data\" -Force
        Write-Host "  ✓ 移动 $($file.Name) 到 data/" -ForegroundColor Green
    }
}

# 创建 __init__.py 文件
Write-Host "`n创建Python包文件..." -ForegroundColor Cyan
$initContent = '"""Urban AI Recommendation System Package"""

__version__ = "1.0.0"
__author__ = "Your Name"
'

@("src", "tests") | ForEach-Object {
    $initPath = Join-Path $_ "__init__.py"
    if (-not (Test-Path $initPath)) {
        Set-Content -Path $initPath -Value $initContent
        Write-Host "  ✓ 创建 $_/__init__.py" -ForegroundColor Green
    }
}

# 创建示例配置文件
Write-Host "`n创建配置文件..." -ForegroundColor Cyan
$configContent = @"
{
    "system": {
        "name": "Urban AI Recommendation System",
        "version": "1.0.0",
        "debug": true
    },
    "recommendation": {
        "default_count": 10,
        "distance_threshold_km": 5,
        "min_rating": 3.0
    },
    "api": {
        "host": "localhost",
        "port": 8000,
        "reload": true
    }
}
"@

$configPath = "config/config.json"
if (-not (Test-Path $configPath)) {
    Set-Content -Path $configPath -Value $configContent
    Write-Host "  ✓ 创建 config/config.json" -ForegroundColor Green
}

# 创建.gitignore文件
Write-Host "`n创建.gitignore..." -ForegroundColor Cyan
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data
data/*.csv
data/*.xlsx
data/temp/

# Config
config/local.json
.env

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent
Write-Host "  ✓ 创建 .gitignore" -ForegroundColor Green

# 创建示例测试文件
Write-Host "`n创建示例测试文件..." -ForegroundColor Cyan
$testContent = @'
"""Urban AI Recommendation System - Unit Tests"""

import unittest
import sys
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestRecommendationSystem(unittest.TestCase):
    """推荐系统测试类"""

    def setUp(self):
        """测试前设置"""
        self.test_data = {
            "users": [{"user_id": "test_user", "preferences": ["food"]}],
            "locations": [{"id": "loc1", "name": "Test Restaurant"}]
        }

    def test_import(self):
        """测试模块导入"""
        try:
            import recommendation_system
            self.assertTrue(True)
        except ImportError:
            self.fail("无法导入recommendation_system模块")

    def test_data_loading(self):
        """测试数据加载"""
        # 这里添加实际的测试代码
        self.assertIsNotNone(self.test_data)

if __name__ == '__main__':
    unittest.main()
'@

$testPath = "tests/test_recommendation_system.py"
Set-Content -Path $testPath -Value $testContent
Write-Host "  ✓ 创建 tests/test_recommendation_system.py" -ForegroundColor Green

# 显示项目结构
Write-Host "`n项目结构:" -ForegroundColor Cyan
Get-ChildItem -Path . -Recurse -Directory | ForEach-Object {
    $indent = "  " * ($_.FullName.Split('\').Count - $projectPath.Split('\').Count - 1)
    Write-Host "$indent📁 $($_.Name)" -ForegroundColor Yellow
}

Get-ChildItem -Path . -Recurse -File | Where-Object { $_.Extension -in @('.py', '.json', '.txt', '.md') } | ForEach-Object {
    $indent = "  " * ($_.DirectoryName.Split('\').Count - $projectPath.Split('\').Count)
    $icon = switch ($_.Extension) {
        '.py' { '🐍' }
        '.json' { '📋' }
        '.txt' { '📄' }
        '.md' { '📝' }
        default { '📄' }
    }
    Write-Host "$indent$icon $($_.Name)" -ForegroundColor White
}

Write-Host "`n✅ 项目设置完成！" -ForegroundColor Green
Write-Host "`n下一步:" -ForegroundColor Yellow
Write-Host "1. 激活虚拟环境: venv\Scripts\activate" -ForegroundColor White
Write-Host "2. 安装依赖: pip install -r requirements.txt" -ForegroundColor White
Write-Host "3. 运行演示: python run_demo.py" -ForegroundColor White