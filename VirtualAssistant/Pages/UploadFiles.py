import streamlit as st
import string

st.header("上传你的文件")


def count_words(text):
    words = text.split()
    return len(words)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

st.title("文件上传学习功能")

uploaded_file = st.file_uploader("请上传一个文本文件", type=["txt"])

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

    # 去除标点符号
    file_contents_no_punctuation = remove_punctuation(file_contents)

    # 统计单词数量
    word_count = count_words(file_contents_no_punctuation)

    st.write(f"文件内容如下：")
    st.write(file_contents)

    st.write(f"去除标点符号后的内容如下：")
    st.write(file_contents_no_punctuation)

    st.write(f"文件中的单词数量为：{word_count}")