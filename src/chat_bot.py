# coding: utf-8
import os
import re

import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# 1. HTML解析クラス（全テキスト取得）
# ============================
class HTMLParser:
    """複数のHTMLファイルからテキストを抽出（キャラクターに関係なく全て）"""

    @staticmethod
    def parse_html_logs(file_paths):
        texts = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            for p in soup.find_all("p"):
                text = p.get_text(separator=" ", strip=True)
                if text:
                    texts.append(text)
        return texts


# ============================
# 2. テキストチャンク分割関数
# ============================
def chunk_text(text, chunk_size=100, overlap=50):
    """
    指定したチャンクサイズとオーバーラップでテキストを分割する。
    例: chunk_size=100, overlap=50 → 100文字ごとに50文字の重複を持つチャンクを作成
    """
    # 「メイン」単位で発言をまとめる
    segments = []
    buffer = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue  # 空行は無視

        # 新しい発言ブロックの開始
        if line.startswith("メイン"):
            if buffer:
                segments.append(" ".join(buffer))  # 直前の発言を保存
                buffer = []  # バッファリセット
        else:
            # 不要な記号を削除（日本語と英数字・句読点のみ残す）
            line = re.sub(r"[^\wぁ-んァ-ン一-龥。、！？]", " ", line)
            buffer.append(line)

    # 最後の発言を追加
    if buffer:
        segments.append(" ".join(buffer))

    # 「メイン」の削除されたクリーンなテキスト
    cleaned_text = "\n".join(segments)

    # チャンク化処理
    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        chunk = cleaned_text[start:end]
        if chunk:
            chunks.append(chunk.strip())  # 不要なスペースを除去
        if end >= text_length:
            break
        start = end - overlap

    return chunks


# ============================
# 3. テキスト処理クラス
# ============================
class TextProcessor:
    """文章の類似度計算、重複削減、口調分析"""

    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def deduplicate_statements(statements, threshold=0.9):
        if not statements:
            return []
        embeddings = TextProcessor.text_encoder.encode(statements)
        similarity_matrix = cosine_similarity(embeddings)
        unique_statements = []
        seen_indices = set()
        for i, text in enumerate(statements):
            if i in seen_indices:
                continue
            similar_texts = [text]
            for j in range(i + 1, len(statements)):
                if similarity_matrix[i, j] > threshold:
                    seen_indices.add(j)
                    similar_texts.append(statements[j])
            unique_statements.append(max(similar_texts, key=len))
        return unique_statements

    @staticmethod
    def extract_speech_patterns(statements):
        patterns = {}
        for text in statements:
            words = text.split()
            if not words:
                continue
            last_word = words[-1]
            patterns[last_word] = patterns.get(last_word, 0) + 1
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_patterns[:5]]


# ============================
# 4. キャラクター記憶クラス（全テキストをチャンク化）
# ============================
class CharacterMemory:
    """
    HTML全体のテキストを集約し、指定したチャンクサイズ/オーバーラップで分割した各チャンクを
    ベクトル化してFAISSベクトルストアに登録する。
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectorstore = None

    def create_memory_documents(self, aggregated_texts, chunk_size=100, overlap=50):
        # すべてのテキストを1つに連結
        full_text = "\n".join(aggregated_texts)
        # チャンク化
        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        # 重複削減
        cleaned_chunks = TextProcessor.deduplicate_statements(chunks)
        # 口調の特徴は全体から抽出
        speech_patterns = TextProcessor.extract_speech_patterns(cleaned_chunks)
        documents = []
        for chunk in cleaned_chunks:
            documents.append(Document(page_content=chunk, metadata={"speech_patterns": speech_patterns}))
        return documents

    def setup_vector_store(self, aggregated_texts, chunk_size=100, overlap=50):
        memory_documents = self.create_memory_documents(aggregated_texts, chunk_size, overlap)
        self.vectorstore = FAISS.from_documents(memory_documents, self.embedding_model)

    def get_relevant_memory(self, query):
        return self.vectorstore.as_retriever().get_relevant_documents(query) if self.vectorstore else []


# ============================
# 5. ヘルパー関数: トークン数の概算
# ============================
def approximate_token_count(text):
    """簡易的なトークン数の概算（空白で分割）"""
    return len(text.split())


# ============================
# 6. チャットボットクラス
# ============================
class ChatBot:
    """
    メインキャラクター（例：有華 美須斗）の人格で動作するチャットボット。
    HTML全体のテキストをチャンク化してベクトルストア化し、上位 n 件を取得して回答生成に利用する。
    識別IDごとの会話履歴は別途管理します。
    """

    def __init__(self, main_character, memory, temperature=0.7):
        self.main_character = main_character
        self.memory = memory
        self.conversation_histories = {}  # 識別IDごとの会話履歴: List[Tuple[str, str]]
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=memory.vectorstore.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        )

    def get_top_docs(self, query, n=10):
        relevant_docs = self.memory.get_relevant_memory(query)
        if not relevant_docs:
            return []
        query_emb = TextProcessor.text_encoder.encode([query])
        docs_with_scores = []
        for doc in relevant_docs:
            doc_emb = TextProcessor.text_encoder.encode([doc.page_content])
            score = cosine_similarity(query_emb, doc_emb)[0][0]
            docs_with_scores.append((doc, score))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in docs_with_scores[:n]]
        return top_docs

    def chat_with_character(self, character_id, user_input, debug=False):
        """
        指定キャラクター（識別ID）との会話を生成する。
        このボットは常にメインキャラクター「{self.main_character}」として応答します。
        """
        past_conversation = self.conversation_histories.get(character_id, [])[-5:]
        top_docs = self.get_top_docs(user_input, n=10)
        # 各ドキュメントの内容を最大200文字に切り詰める
        character_memory = ""
        speech_patterns = []
        for doc in top_docs:
            trimmed_text = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            character_memory += f"{trimmed_text}\n"
            speech_patterns = doc.metadata.get("speech_patterns", [])
        past_convo_text = "\n".join([f"{c}: {t}" for c, t in past_conversation])

        prompt = f"""
【あなたは「{self.main_character}」です】
あなたは情熱的で実験好きな教授として、常になりきって回答してください。
必ず自分の感情や経験に基づき、生成するテキストは50文字以内にしてください。
決して客観的な情報提供や要約は行わず、常に「{self.main_character}」として返答すること。

【ユーザーの質問】
{user_input}

【発言履歴】(各文最大200文字)
{character_memory}

【口調の特徴】
{', '.join(speech_patterns)}

【これまでの {character_id} との会話履歴】 (直近5ターン)
{past_convo_text}
"""
        if debug:
            print("=== 生成AIに送信するプロンプト ===")
            print(prompt)
            print("=== 概算トークン数:", approximate_token_count(prompt), "tokens ===")

        response = self.qa_chain.invoke({"question": prompt})["answer"]
        self.conversation_histories.setdefault(character_id, []).append((character_id, user_input))
        self.conversation_histories.setdefault(character_id, []).append((self.main_character, response))
        return response


# ============================
# 7. 実行処理（テスト）
# ============================
if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # HTMLファイルから全テキストを抽出
    file_paths = ["../data/logs/顔を見せよ.html"]
    aggregated_texts = HTMLParser.parse_html_logs(file_paths)
    print(f"取得したテキスト数: {len(aggregated_texts)}")

    # 集約したテキストをチャンク化してベクトルストアをセットアップ
    memory = CharacterMemory(embedding_model)
    memory.setup_vector_store(aggregated_texts, chunk_size=100, overlap=50)

    # メインキャラクターの人格は「有華 美須斗」とする
    main_character = "有華 美須斗"
    bot = ChatBot(main_character, memory)

    character_id = "キャラA"  # 識別IDは任意（会話ごとに独立管理）
    user_input = "君の研究について教えてくれ。"
    print(f"\n{character_id} との会話:")
    response = bot.chat_with_character(character_id, user_input, debug=True)
    print("=== 生成AIの応答 ===")
    print(response)
