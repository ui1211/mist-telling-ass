import glob
import os
from pprint import pprint

import google.generativeai as genai
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

from src.log_parser import character_logs_from_files, extract_character_features

GOOGLE_API_KEY = "AIzaSyBNt1EhID_8b4WJiodrk2jrmd17G2q81U8"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(initial_sidebar_state="collapsed")


class App:

    def __init__(self, model):
        self.model = model

        if "character_logs" not in st.session_state:
            html_file_paths = glob.glob("data/logs/*")
            st.session_state["character_logs"], _ = character_logs_from_files(html_file_paths)
            st.session_state["chat_history"] = []

    def chat_with_gemini(
        self,
        character_name,
        character_features,
        character_memo,
        user_name,
        user_input,
        chat_history=None,
        current_situation=None,
        situation_history=None,
        debug=False,
    ):
        prompt = f"""
        あなたの名前は「{character_name}」です。
        質問者名前に対して質問内容を踏まえて自然な文章で回答してください。
        現在の状況や過去の状況の推移から回答を生成してください。

        ---質問者情報---
        質問者名前: {user_name}
        質問内容: {user_input}

        ---生成ルール---
        生成する文字数は100文字以内に設定してください。
        過去の発言履歴からキャラクターの特徴を踏襲してください。

        ---しゃべり方の特徴---
        あなたのしゃべり方の特徴は以下の通りです。単語とその出現回数のペアになります。
        無理に使用する必要はありませんが、参考にしてください。
        {character_features}

        ---略歴---
        {character_memo}

        ---過去の発言履歴---
        {chat_history}

        ---現在の状況---
        {current_situation}

        --状況の推移--
        {situation_history}
        """

        if debug:
            print(prompt)

        response = self.model.generate_content(prompt)
        return response.text

    def layout(self):

        st.image("data/logo/logo.png", width=800)

        character_logs = st.session_state["character_logs"]

        with st.sidebar:
            self.character_name = st.selectbox(
                label="Character",
                options=list(character_logs.keys()),
                index=5,
            )

            self.character_features = extract_character_features(
                character_logs[self.character_name], most_common=5, min_count=5
            )

            self.character_memo = st.text_area(
                label="Character Memo",
                value="""錬金術に魅入られた大学教授\n世界の物質は第一物質であるエーテルから構成されていると信じて日夜錬金術の研究を行っている。\n研究の関係から様々な学問に通じており大学では化学について教鞭をふるっている。""",
                height=200,
            )

            self.user_name = st.text_input(label="User Name", value="")

            self.current_situation = st.text_area(
                label="Current Situation",
                value="",
                height=100,
            )

        self.user_input = st.chat_input("Say something")

        if self.user_input:
            self.response = self.chat_with_gemini(
                self.character_name,
                self.character_features,
                self.character_memo,
                self.user_name,
                self.user_input,
                chat_history=st.session_state["chat_history"],
            )

            # チャット履歴に追加
            st.session_state["chat_history"].append(
                {
                    "user_input": self.user_input,
                    "response": self.response,
                }
            )

        # 吹き出し表示
        for item in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.write(item["user_input"])
            with st.chat_message("assistant"):
                st.write(item["response"])

    def main(self):
        if "character_logs" in st.session_state:
            self.layout()


if __name__ == "__main__":

    app = App(model)
    app.main()
