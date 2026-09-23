from src.generation.generate_response import generate_response
from src.retrieval.retrieve_chunks import retrieve_chunks


def main():
    print("""
    I'd love to help you build a new adventure! To get us started, where would you like to go?
    How would you like to plan your trip to California National Parks?
    """)
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break

        relevant_chunks = retrieve_chunks(user_input)

        if not relevant_chunks:
            print("I could not find relevant information for your question.")
            continue
        response = generate_response(user_input, relevant_chunks)
        print(response)

main()