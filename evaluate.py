import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from app import load_vectorstore, get_answer

load_dotenv()


def prepare_test_data():
    test_queries = [
        {
            "question": "What are the objectives of robotics?",
            "ground_truth": "Robotics teaches kinematics, dynamics, and control systems.",
        },
        {
            "question": "List the modules in AI curriculum",
            "ground_truth": "Modules include Machine Learning, Deep Learning, NLP, and Computer Vision.",
        },
        {
            "question": "What is the syllabus structure?",
            "ground_truth": "The syllabus is organized by units, course objectives, and modules.",
        },
    ]
    return test_queries


def run_rag_pipeline(query, vectorstore):
    answer, sources = get_answer(query, vectorstore)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    retrieved_docs = retriever.invoke(query)
    
    context_list = [doc.page_content for doc in retrieved_docs[:40]]
    
    return answer, context_list


def create_evaluation_dataset(test_queries, vectorstore):
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    for test in test_queries:
        question = test["question"]
        ground_truth = test["ground_truth"]
        
        answer, contexts = run_rag_pipeline(question, vectorstore)
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    
    return dataset


def evaluate_rag_system(test_queries, vectorstore):
    dataset = create_evaluation_dataset(test_queries, vectorstore)
    
    evaluation_scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    return evaluation_scores


def display_results(scores):
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, score_value in scores.items():
        print(f"{metric_name}: {score_value:.4f}")
    
    print("="*50 + "\n")


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
