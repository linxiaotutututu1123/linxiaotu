# 量化交易系统打包脚本
# Quant Trading System Build Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "量化交易系统 - 打包脚本" -ForegroundColor Cyan
Write-Host "Quant Trading System - Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 切换到脚本所在目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# 清理旧的构建文件
Write-Host "[1/5] 清理旧的构建文件..." -ForegroundColor Yellow
$foldersToRemove = @("build", "dist", "*.egg-info")
foreach ($folder in $foldersToRemove) {
    if (Test-Path $folder) {
        Remove-Item -Recurse -Force $folder
        Write-Host "  已删除: $folder" -ForegroundColor Gray
    }
}

# 清理缓存
Get-ChildItem -Path . -Include "__pycache__" -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include ".pytest_cache" -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include "*.pyc" -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "  已清理缓存文件" -ForegroundColor Gray
Write-Host ""

# 检查Python环境
Write-Host "[2/5] 检查Python环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "  Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  错误: 未找到Python" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 安装构建工具
Write-Host "[3/5] 安装构建工具..." -ForegroundColor Yellow
pip install --upgrade pip build wheel setuptools -q
Write-Host "  构建工具已准备就绪" -ForegroundColor Green
Write-Host ""

# 构建包
Write-Host "[4/5] 构建分发包..." -ForegroundColor Yellow
python -m build
if ($LASTEXITCODE -eq 0) {
    Write-Host "  构建成功!" -ForegroundColor Green
} else {
    Write-Host "  构建失败!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 显示结果
Write-Host "[5/5] 构建结果..." -ForegroundColor Yellow
Write-Host ""
Write-Host "生成的文件:" -ForegroundColor Cyan
Get-ChildItem -Path "dist" | ForEach-Object {
    $size = "{0:N2} MB" -f ($_.Length / 1MB)
    Write-Host "  $($_.Name) ($size)" -ForegroundColor White
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "打包完成!" -ForegroundColor Green
Write-Host ""
Write-Host "安装方法:" -ForegroundColor Yellow
Write-Host "  pip install dist\quant_trading_system-1.0.0-py3-none-any.whl" -ForegroundColor White
Write-Host ""
Write-Host "或使用源码包:" -ForegroundColor Yellow
Write-Host "  pip install dist\quant_trading_system-1.0.0.tar.gz" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
