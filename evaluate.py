import os
import sys
sys.path.append("backend")

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app import load_vectorstore, get_answer  # type: ignore

load_dotenv()


def prepare_test_data():
    return [
        {
            "question": "What are the course outcomes of Drive System for Robotics?",
            "ground_truth": "copy the actual CO text from your PDF here",
        },
        {
            "question": "What is covered in Module 3 of Drive System for Robotics?",
            "ground_truth": "copy the actual module 3 content from your PDF here",
        },
        {
            "question": "What is the syllabus structure for 5th semester?",
            "ground_truth": "copy the actual syllabus structure from your PDF here",
        },
    ]


def run_rag_pipeline(query, vectorstore):
    answer, sources = get_answer(query, vectorstore)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    retrieved_docs = retriever.invoke(query)
    context_list = [doc.page_content for doc in retrieved_docs[:40]]
    return answer, context_list


def create_evaluation_dataset(test_queries, vectorstore):
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for test in test_queries:
        answer, contexts = run_rag_pipeline(test["question"], vectorstore)
        questions.append(test["question"])
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(test["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def evaluate_rag_system(test_queries, vectorstore):
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    langchain_llm = LangchainLLMWrapper(gemini_llm)
    langchain_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

    dataset = create_evaluation_dataset(test_queries, vectorstore)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=langchain_llm,
        embeddings=langchain_embeddings,
    )
    return scores


def display_results(scores):
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric_name, score_value in scores.items():
        print(f"{metric_name}: {score_value:.4f}")
    print("=" * 50 + "\n")


def main():
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("Preparing test queries...")
    test_queries = prepare_test_data()

    print(f"Running evaluation on {len(test_queries)} queries...")
    scores = evaluate_rag_system(test_queries, vectorstore)

    display_results(scores)


if __name__ == "__main__":
    main()