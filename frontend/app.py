import os
from typing import Any, Dict, List

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="BioMed Research Agent Demo", layout="wide")

st.title("BioMed Research Agent Demo")

st.sidebar.header("连接配置")
backend_url = st.sidebar.text_input("Backend URL", value=BACKEND_URL)

st.sidebar.markdown("""
**功能**
- 上传PDF/Markdown
- 构建FAISS向量索引
- RAG问答 + 引用展示
- 生成POC交付报告
""")


st.header("1) 上传资料")
uploaded_files = st.file_uploader(
    "选择PDF/Markdown/TXT文件", type=["pdf", "md", "markdown", "txt"], accept_multiple_files=True
)

if st.button("上传到后端"):
    if not uploaded_files:
        st.warning("请先选择文件")
    else:
        for file in uploaded_files:
            response = requests.post(
                f"{backend_url}/upload",
                files={"file": (file.name, file.getvalue())},
                timeout=60,
            )
            if response.ok:
                data = response.json()
                st.success(f"上传成功: {data['filename']} -> {data['file_id']}")
            else:
                st.error(f"上传失败: {file.name} ({response.text})")

st.header("2) 构建向量索引")
if st.button("开始索引"):
    response = requests.post(f"{backend_url}/index", json={}, timeout=120)
    if response.ok:
        data = response.json()
        st.success(f"索引完成: {data['chunks']} chunks")
        st.write(data)
    else:
        st.error(f"索引失败: {response.text}")

st.header("3) RAG问答")
question = st.text_input("请输入问题")
if st.button("提交问题"):
    if not question:
        st.warning("请填写问题")
    else:
        response = requests.post(
            f"{backend_url}/ask", json={"question": question, "top_k": 4}, timeout=120
        )
        if response.ok:
            data = response.json()
            st.subheader("回答")
            st.write(data["answer"])
            st.subheader("引用")
            for citation in data["citations"]:
                st.markdown(
                    f"**{citation['source']}** (chunk {citation['chunk_id']})\n\n> {citation['text']}"
                )
        else:
            st.error(f"请求失败: {response.text}")

st.header("4) 生成POC交付报告")
if st.button("生成报告"):
    if not question:
        st.warning("请先输入问题")
    else:
        response = requests.post(
            f"{backend_url}/agent/run",
            json={"question": question, "action": "poc_report"},
            timeout=120,
        )
        if response.ok:
            data = response.json()
            report = data.get("report_markdown") or ""
            st.subheader("报告预览")
            st.markdown(report)
            st.download_button(
                "下载报告", data=report, file_name="poc_report.md", mime="text/markdown"
            )
        else:
            st.error(f"生成失败: {response.text}")
