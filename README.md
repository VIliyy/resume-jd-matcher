# Resume × JD Matcher

> AI 驱动的简历-岗位匹配优化工具 —— 输入你的简历 + 目标 JD，输出优化后的简历。

## 功能

- 🔑 DeepSeek API 驱动，智能匹配简历与岗位描述
- 📋 自动识别简历模块（教育/实习/项目/技能），分区优化
- 📊 输出末尾附带 JD 匹配度评分
- 💾 本地自动保存历史记录（`history/` 目录）
- 🎨 taste-skill 风格前端，深色主题 + 珊瑚红 accent

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python -X utf8 app.py

# 3. 浏览器打开
http://127.0.0.1:5000
```

## 使用方式

1. 输入你的 **DeepSeek API Key**
2. 粘贴 **原始简历** 文本
3. 粘贴 **目标岗位 JD** 文本
4. 点击「开始匹配优化」
5. 复制优化结果，直接用于投递

## 项目结构

```
├── app.py              # Flask 后端 + DeepSeek API + Prompt 工程
├── requirements.txt    # flask, requests
├── templates/
│   └── index.html      # 前端页面
├── history/            # 历史记录（已 gitignore）
└── .gitignore
```

## 依赖

- Python ≥ 3.9
- Flask ≥ 3.0
- requests ≥ 2.31
- DeepSeek API Key（从 [platform.deepseek.com](https://platform.deepseek.com) 获取）

## License

MIT
