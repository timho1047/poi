from datasets import Dataset
from poi.llm import LLMConfig, train_llm

toy_data = [
    {
        "instruction": "What is 15 + 27?",
        "response": "Let me help you solve this step by step:\n15 + 27 = 42\n\nTo break it down: 15 + 20 = 35, then 35 + 7 = 42.",
    },
    {
        "instruction": "Explain what a prime number is.",
        "response": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. For example, 2, 3, 5, 7, 11 are prime numbers. The number 4 is not prime because it can be divided by 2.",
    },
    {
        "instruction": "How do I calculate the area of a circle?",
        "response": "To calculate the area of a circle, use the formula:\n\nArea = π × r²\n\nWhere r is the radius of the circle and π (pi) is approximately 3.14159. For example, if the radius is 5, the area would be π × 5² = π × 25 ≈ 78.54 square units.",
    },
    {
        "instruction": "What is 8 × 7?",
        "response": "8 × 7 = 56\n\nYou can think of this as adding 8 seven times: 8 + 8 + 8 + 8 + 8 + 8 + 8 = 56",
    },
    {
        "instruction": "What is the Pythagorean theorem?",
        "response": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (the longest side) equals the sum of squares of the other two sides.\n\nFormula: a² + b² = c²\n\nWhere c is the hypotenuse and a and b are the other two sides.",
    },
    {
        "instruction": "How do you find the mean of a set of numbers?",
        "response": "To find the mean (average) of a set of numbers:\n\n1. Add all the numbers together\n2. Divide by how many numbers there are\n\nExample: For [4, 8, 6, 5, 10]\nMean = (4 + 8 + 6 + 5 + 10) ÷ 5 = 33 ÷ 5 = 6.6",
    },
    {
        "instruction": "What is 100 - 37?",
        "response": "100 - 37 = 63\n\nOne way to solve this: 100 - 30 = 70, then 70 - 7 = 63",
    },
    {
        "instruction": "Explain what a fraction is.",
        "response": "A fraction represents a part of a whole. It consists of two numbers:\n\n- Numerator (top number): tells you how many parts you have\n- Denominator (bottom number): tells you how many parts make up the whole\n\nFor example, 3/4 means you have 3 parts out of 4 total parts.",
    },
]

toy_eval_data = [
    {
        "instruction": "How do you find the mean of a set of numbers?",
        "response": "To find the mean (average) of a set of numbers:\n\n1. Add all the numbers together\n2. Divide by how many numbers there are\n\nExample: For [4, 8, 6, 5, 10]\nMean = (4 + 8 + 6 + 5 + 10) ÷ 5 = 33 ÷ 5 = 6.6",
    },
    {
        "instruction": "What is 100 - 37?",
        "response": "100 - 37 = 63\n\nOne way to solve this: 100 - 30 = 70, then 70 - 7 = 63",
    },
    {
        "instruction": "Explain what a fraction is.",
        "response": "A fraction represents a part of a whole. It consists of two numbers:\n\n- Numerator (top number): tells you how many parts you have\n- Denominator (bottom number): tells you how many parts make up the whole\n\nFor example, 3/4 means you have 3 parts out of 4 total parts.",
    },
]


# Convert to Llama-3 chat format
def format_chat_template(example):
    """Format the data in Llama-3 instruction format"""
    text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example["instruction"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example["response"]}<|eot_id|>"""
    return {"text": text}


# Create dataset
train_dataset = Dataset.from_list(toy_data)
eval_dataset = Dataset.from_list(toy_eval_data)
train_dataset = train_dataset.map(format_chat_template)
eval_dataset = eval_dataset.map(format_chat_template)

print(f"Train dataset size: {len(train_dataset)} examples")
print(f"Eval dataset size: {len(eval_dataset)} examples")


if __name__ == "__main__":
    config = LLMConfig()
    train_llm(config, train_dataset, eval_dataset)
