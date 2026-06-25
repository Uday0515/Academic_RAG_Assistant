import os
import time
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

        # ─────────────────────────────────────────
        # 3RD SEMESTER
        # ─────────────────────────────────────────

        {
            "question": "What are the course outcomes of Drive Systems for Robotics?",
            "ground_truth": "CO1: Understand basic knowledge of different types of drive systems with respect to robot motion analysis. CO2: Interpret the hydraulic drive mechanism, circuits and its considerations. CO3: Familiarize the pneumatic drive mechanism, circuit designs and its advantages. CO4: Understand the importance of electrical drive systems and implementations of electro pneumatic drive. CO5: Know the electrical drive characteristics and familiarize about advanced drives like PLC, its applications, and supervisory control systems."
        },
        {
            "question": "What is covered in Module 3 of Drive Systems for Robotics?",
            "ground_truth": "Module 3 covers Pneumatic Drives: Introduction to pneumatics system, choice of working medium, components of pneumatic power systems, production of compressed air, compressor-rotary vane compressor, air pressure regulator, pneumatic actuators, classification of air cylinders, construction of linear cylinder (double acting), rotary actuators-vane and turbine type air motors, design parameters, pneumatic control valves classification, direction control valves (DCV), flow control valves (FCV), pressure control valve, filter regulator lubricator, design of pneumatic circuits including direct actuation, indirect actuation, and principle of cascade systems. Advantages, disadvantages and applications of pneumatic drives are also covered."
        },

        # ─────────────────────────────────────────
        # 4TH SEMESTER
        # ─────────────────────────────────────────

        {
            "question": "What are the course outcomes of Applied Artificial Intelligence?",
            "ground_truth": "CO1: Evaluate Artificial Intelligence methods and describe their foundations. CO2: Apply basic principles of AI in solutions that require problem-solving, inference, perception, knowledge representation, and learning. CO3: Demonstrate knowledge of reasoning and knowledge representation for solving real-world problems. CO4: Analyze and illustrate how search algorithms play a vital role in problem solving. CO5: Illustrate the construction of learning and expert systems."
        },
        {
            "question": "What is covered in Module 5 of Probability Theory and Optimization?",
            "ground_truth": "Module 5 covers Reliability: Definition, bathtub curve, causes of failures, and various phases in equipment life. Component Reliability and Hazard Model: Introduction, component reliability from test data, difference between failure rate and hazard rate, time dependent hazard models, Mean Time Between Failures (MTBF), and Mean Time to Failures (MTTF) with application problems. Self-study component includes problems on probability of failure."
        },

        # ─────────────────────────────────────────
        # 5TH SEMESTER
        # ─────────────────────────────────────────

        {
            "question": "What are the course outcomes of Robot Kinematics and Dynamics (22RI51)?",
            "ground_truth": "CO1: Explain position and orientation parameters for describing the pose of industrial robots. CO2: Apply mathematical tools for solving robot kinematics problems. CO3: Assign the coordinate frames to industrial robots and derive their forward and inverse kinematic equations. CO4: Use software tools for obtaining solutions to forward and inverse kinematics problems."
        },
        {
            "question": "What is covered in Module 1 of Robot Kinematics and Dynamics?",
            "ground_truth": "Module 1 (Introduction) covers: Introduction to Robotics, Elements of Robots including joints, links, end effectors, grippers, actuators and sensors, Fundamentals of Robot Degrees of Freedom, Robot Components, Rigid body motions, Concepts of Rigid Body, Robotic manipulator Frames, Euclidean Space, Inertial Frame, Fundamentals of Robotic Manipulator, Vectors and Matrices. It maps to CO1."
        },

        # ─────────────────────────────────────────
        # 6TH SEMESTER
        # ─────────────────────────────────────────

        {
            "question": "What are the course outcomes of Embedded System Design (22RI61)?",
            "ground_truth": "CO1: Understand basic embedded system concepts, architectures, and hardware components. CO2: Develop microcontroller programs using GPIO, timers, PWM, ADC/DAC, serial interfaces, and interrupts. CO3: Apply real-time system concepts and FreeRTOS for multitasking and synchronization. CO4: Implement communication protocols and integrate embedded systems with robotic subsystems. CO5: Design and demonstrate an embedded project using sensors, actuators, and AI inference modules."
        },
        {
            "question": "What is covered in Module 1 of Embedded System Design?",
            "ground_truth": "Module 1 (Fundamentals of Embedded Systems) covers: Definition and characteristics of embedded systems, Von Neumann and Harvard architectures, Microcontroller vs Microprocessor, Basic hardware components including CPU, memory and peripherals, Digital I/O concepts, Sensors and actuators overview, Introduction to development boards including Arduino and Pico, Embedded workflow, and Simple programs using GPIO. Maps to CO1."
        },

    ]

def run_rag_pipeline(query, vectorstore):
    time.sleep(15)
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