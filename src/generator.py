from together import Together



def generate_answer(query,top_answers):
    client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")

    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages = [
        {
            "role": "user",
            "content": f"""You are a question answering assistant. The user will be asking a query. Based on the query, the context is already retrieved. So you job is to go through the context given and frame the answer for the user as per his query. Remember, the answer should be solely based on the context given. The query and context are given below:
            
            user query : {query}
            retrieved context: {top_answers}


            Make sure you are confident enough while answering. Assume you know everything from the context and directly start with the answer without any assumptions.
            
            """
        }
    ]
    )
    print(response.choices[0].message.content)
    return(response.choices[0].message.content)