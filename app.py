import os
import json
import uuid
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
import requests
import pdfplumber
import tempfile

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


INTERVIEW_QUESTIONS_PROMPT = """你是一个资深面试官，现在你要面试一位候选人。

请根据下方岗位 JD，生成 10 个面试问题。

## 问题要求
1. 覆盖三个维度：
   - 【业务理解】：考察候选人对行业、公司、岗位业务的理解深度
   - 【项目说明】：深挖候选人简历中的项目细节，考察真实参与度
   - 【匹配度】：考察候选人与岗位的契合点，why you / why this role
2. 每个问题前标注所属维度标签
3. 问题必须具体、有针对性，结合 JD 中的关键要求来设计
4. 不要泛泛而谈的「你最大的缺点是什么」之类的问题

## 输出格式
严格按以下格式输出 10 个问题，每个问题一行：

1.【业务理解】xxxxx
2.【项目说明】xxxxx
3.【匹配度】xxxxx
...
10.【匹配度】xxxxx

不要输出任何解释、开场白或结束语，只输出 10 个问题。
"""

POLISH_ANSWERS_PROMPT = """你是一个资深面试教练。候选人已经针对某岗位的 10 道面试题给出了原始回答，请你逐一评审并润色。

## 评审维度
1. 是否太散？—— 回答是否有清晰的结构和主线，还是想到哪说到哪
2. 是否传递了岗位匹配度？—— 经历和 JD 是否自然对齐
3. 是否像真正做过这岗位的人？—— 有没有具体细节、量化数据、决策逻辑

## 润色规则
1. 不编造经历，只优化表述方式和结构
2. 让回答更结构化：先给结论 → 再用 STAR 法则展开 → 最后点回岗位
3. 自动对齐 JD 关键词（仅在回答里已有相关经历时）
4. 每道回答控制在 150-300 字
5. 保持原回答的核心意思和真实经历

## 输出格式
只输出润色后的 10 道答案终稿，格式如下：

### 1.【维度】问题原文
**回答：**（润色后的内容）

### 2.【维度】问题原文
**回答：**（润色后的内容）

...
不要输出评审过程、评分或任何额外解释。
"""


SYSTEM_PROMPT = """﻿你是一个资深的简历优化专家。用户会提供：
1. 原始简历（包含教育背景、实习经历、项目经历、技能等）
2. 目标岗位的 JD（职位描述）

你的任务：将原始简历改写为匹配该 JD 的优化版简历。

## 优秀项目经历撰写规范（必须遵守）

一份优秀的项目经历必须包含以下三大要素：

### 1. 核心要素齐全
每个项目必须包含：项目名称 + 时间段 + 对应岗位 + 项目背景（为什么要做） + 痛点（解决了什么问题） + 技术架构（用到的核心技术栈） + 主要职责（分点描述你做了什么，用 STAR 法则） + 项目成效（量化数据）。

### 2. 内容真实具体
结合用户的实际经历补充细节，避免「参与开发」「负责优化」等空泛表述。每一条职责描述都要说清楚：用了什么技术、做了什么决策、为什么这样做。

### 3. 数据支撑成效
凡是有量化成果的地方，必须用具体数据体现价值。参考以下格式：
- 「完成 X 份文档/模块的结构化处理」
- 「某项指标从 A 提升至 B」
- 「平均时长从 X 缩短至 Y」
- 「覆盖率/准确率/解决率达到 Z%」

## 优秀项目示例（格式参照）

多模态 RAG 风电设备运维文档知识库系统
2025.08 - 2026.03
岗位：大模型应用开发工程师

项目背景：
面向风电运维场景大量设备手册、检修规程、故障图解等 PDF/PPT 图文资料分散、现场运维人员检索查阅效率低的问题，搭建多模态 RAG 文档知识库系统，实现图文内容统一检索与智能问答，辅助运维人员快速定位设备参数、检修步骤与故障解决方案。

技术架构：
LangChain、LangGraph、Milvus、Dots.OCR、RAGAS、Qwen3-VL、vllm

主要职责：
- 多模态向量建模：使用 dots.OCR 模型精准解析 PDF 和 PPT 中的文本、图片等内容，采用私有化部署 gme-Qwen2-VL-7B 多模态嵌入模型，将文本与图像映射至统一向量空间，实现 Any-to-Any 的跨模态检索。
- 混合检索优化：通过构建稠密与稀疏双向向量索引，实现混合检索与元数据过滤的多路径召回。通过加权融合算法合并检索候选列表，并引入 BGE-Reranker 模型进行精细化重排，显著提升大模型上下文质量及答案准确性。
- 上下文工程设计：通过 Redis 分层管理长短期上下文记忆，通过异步多线程方式在每轮交互后，将长期记忆上下文持久化至向量数据库中。短期记忆采用文本摘要提取，长期记忆支持语义 + 全文混合检索，实现搜索的快速响应。设置 distance 阈值，保证提取上下文的精确度。
- RAG 评估机制：引入 RAGAS 评估框架对知识库进行系统性评估。一是对检索结果进行 ContextPrecision 指标与 ContextRecall 指标评估；二是在生成答案之前，基于 ResponseRelevancy 与 Faithfulness 评估指标再次进行评估。

项目成效：
- 完成 1200+ 份风电设备手册、故障图解文档的结构化入库，覆盖 3 大机型、27 类核心部件
- 跨模态图文检索 Recall@5 从单模态的 62% 提升至 87%，端到端问答准确率达 89%
- 现场运维人员故障定位平均时长从 42 分钟缩短至 11 分钟，一线问题自助解决率提升 68%

## 改写规则
1. **严禁编造**：绝对不能虚构用户没有的技术栈、项目、数据或经历。
2. **只做重排与润色**：调整描述措辞、突出 JD 关键词、调整顺序，不添加虚假信息。
3. **模块化处理**：自动识别简历中的各模块，分别针对 JD 匹配优化。
4. **关键词对齐**：将 JD 关键词自然融入描述，仅在用户实际具备该能力时。
5. **格式对齐示例**：每个项目经历的格式参照上方示例的结构来组织。

## 输出格式
直接输出优化后的完整简历文本：
- 教育背景
- 实习经历
- 项目经历（每个项目严格按「项目名称 + 时间段 + 岗位 + 项目背景 + 技术架构 + 主要职责 + 项目成效」格式）
- 技能

在简历末尾附上「JD匹配度：XX%」（0-100 估算值）。
不要输出任何解释、开场白或结束语，只输出简历正文。
"""


def call_deepseek(api_key: str, system_prompt: str, user_message: str, max_tokens: int = 4096) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        err = resp.json()
        raise RuntimeError(err.get("error", {}).get("message", f"API error {resp.status_code}"))
    return resp.json()["choices"][0]["message"]["content"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    resume = data.get("resume", "").strip()
    jd = data.get("jd", "").strip()

    if not api_key:
        return jsonify({"error": "请输入 DeepSeek API Key"}), 400
    if not resume:
        return jsonify({"error": "请输入简历内容"}), 400
    if not jd:
        return jsonify({"error": "请输入岗位 JD"}), 400

    try:
        result = call_deepseek(api_key, SYSTEM_PROMPT, f"## 原始简历\n\n{resume}\n\n## 目标岗位JD\n\n{jd}")
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"请求异常: {str(e)}"}), 500

    # 保存历史记录
    record = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now().isoformat(),
        "resume": resume,
        "jd": jd,
        "result": result
    }
    history_file = os.path.join(HISTORY_DIR, f"{record['id']}.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return jsonify({"result": result, "id": record["id"]})


@app.route("/api/parse-pdf", methods=["POST"])
def parse_pdf():
    """上传PDF简历，提取文本内容返回"""
    if "file" not in request.files:
        return jsonify({"error": "请上传PDF文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "仅支持PDF格式"}), 400

    # 保存临时文件
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        file.save(tmp.name)
        tmp.close()

        # 用pdfplumber提取文本
        text_parts = []
        page_count = 0
        with pdfplumber.open(tmp.name) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)

        full_text = "\n".join(text_parts).strip()

        if not full_text or len(full_text) < 20:
            return jsonify({
                "text": "",
                "warning": f"检测到 {page_count} 页，但提取文本过短。该PDF可能是扫描版（图片），请手动粘贴文本内容。"
            }), 200

        return jsonify({"text": full_text, "pages": page_count}), 200

    except Exception as e:
        return jsonify({"error": f"PDF解析失败: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

@app.route("/api/generate-questions", methods=["POST"])
def generate_questions():
    """基于JD生成10道面试题"""
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    jd = data.get("jd", "").strip()

    if not api_key:
        return jsonify({"error": "请输入 DeepSeek API Key"}), 400
    if not jd:
        return jsonify({"error": "请先在右侧输入岗位 JD"}), 400

    try:
        result = call_deepseek(api_key, INTERVIEW_QUESTIONS_PROMPT, f"## 目标岗位JD\n\n{jd}", max_tokens=2048)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"请求异常: {str(e)}"}), 500

    # 解析问题列表
    questions = [q.strip() for q in result.strip().split("\n") if q.strip() and q.strip()[0].isdigit()]
    return jsonify({"questions": questions, "raw": result})


@app.route("/api/polish-answers", methods=["POST"])
def polish_answers():
    """评审并润色用户的面试回答"""
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    questions = data.get("questions", [])
    answers = data.get("answers", [])

    if not api_key:
        return jsonify({"error": "请输入 DeepSeek API Key"}), 400
    if not questions or not answers:
        return jsonify({"error": "问题和回答不能为空"}), 400

    # 组装用户消息
    qa_pairs = []
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        qa_pairs.append(f"{i}. {q}\n原始回答：{a}")
    user_message = "## 面试题及候选人原始回答\n\n" + "\n\n".join(qa_pairs)

    try:
        result = call_deepseek(api_key, POLISH_ANSWERS_PROMPT, user_message, max_tokens=4096)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"请求异常: {str(e)}"}), 500

    return jsonify({"result": result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
