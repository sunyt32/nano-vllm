import glob
import random
from tqdm import tqdm

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config

GENERATE_LENGTH = 20

RANDOM_NEEDLE_CITIES  = [
    'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
    'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
    'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
    'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
    'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
    'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
    'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
    'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
    'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
    'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
    'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
    'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
]

PROMPT_STRUCTURES = {
    "default": 
        lambda sample: f"""\
        There are special magic numbers inside a lot of irrelevant text. Find them and memorize them. I will quiz you about a magic number there.
        {sample['context']}
        {sample['retrieval_question']}
        The special magic {sample['city']} number is:""",
}

def generate_base_ctx(tokenizer, base_ctx, max_ctx_size, limit):
    random.seed(42)
    for _ in tqdm(range(2 * limit), desc="Generating base ctx"):
        file_paths = sorted(glob.glob('/mnt/msranlp/tianzhu/eval/data/PaulGrahamEssays/*.txt'))
        random.shuffle(file_paths)

        context = ""
        ctx_size = 0
        while ctx_size < max_ctx_size:
            for file_path in file_paths:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                ctx_size += len(tokenizer.encode(file_content, add_special_tokens=False))
                context += file_content
                if ctx_size > max_ctx_size:
                    break
        
        base_ctx.append(context)


def trim(tokenizer, base_ctx, ctx_size):
    trimmed_base_ctx = []
    for ctx in base_ctx:
        tokens = tokenizer.encode(ctx, add_special_tokens=False)
        if len(tokens) > ctx_size:
            context = tokenizer.decode(tokens[:ctx_size], skip_special_tokens=True)
        trimmed_base_ctx.append(context)
    return trimmed_base_ctx
    

def insert_single_needle(tokenizer, tokens_context, tokens_needle, ratio):
    if int(ratio) == 1:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * ratio)
        tokens_new_context = tokens_context[:insertion_point]
        period_tokens = tokenizer.encode('cat.')[-1:]
        while len(tokens_new_context) > 0 and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        tokens_new_context = tokens_new_context + tokens_needle + tokens_context[insertion_point:]
    return tokens_new_context
        
    
def insert_needle(tokenizer, trimmed_base_ctx, needles, ctx_size, ratio, distractor_ratios):
    num_needles = len(needles)
    needle_ctx = []
    ctx_size -= 200
    
    for i, ctx in enumerate(trimmed_base_ctx):
        tokens_needles = [tokenizer.encode(needle, add_special_tokens=False)[1:] for needle in needles]
        tokens_context = tokenizer.encode(ctx, add_special_tokens=False)[1:]

        if len(tokens_context) + num_needles * len(tokens_needles[0]) > ctx_size:
            tokens_context = tokens_context[:ctx_size - num_needles * len(tokens_needles[0])]
        
        for tokens_needle, distractor_ratio in zip(tokens_needles[1:], distractor_ratios[i]):
            tokens_context = insert_single_needle(tokenizer, tokens_context, tokens_needle, distractor_ratio)
        tokens_context = insert_single_needle(tokenizer, tokens_context, tokens_needles[0], ratio)
        needle_ctx.append(tokenizer.decode(tokens_context, skip_special_tokens=True))

    return needle_ctx


def insert_prompt(needle_ctx, retrieval_question, target_city):
    prompt_needle_ctx = []
    for ctx in needle_ctx:
        prompt_needle_ctx.append(PROMPT_STRUCTURES["default"]({"context": ctx, 
                                                                "retrieval_question": retrieval_question,
                                                                "city": target_city}))
    return prompt_needle_ctx
    

def model_generation(model, ratio, prompt_needle_ctx, answer, limit):
    prompt_needle_ctx = prompt_needle_ctx[:limit]
    outputs = model.generate(prompt_needle_ctx, SamplingParams(temperature=0.0, max_tokens=GENERATE_LENGTH))
    acc = sum(answer in output['text'] for output in outputs)
    final_acc = acc / len(outputs)
    print([item['text'] for item in outputs])
    print("Eval on: ", len(outputs), " samples")
    print("Ratio", ratio)
    print("Acc: ", final_acc)
    return final_acc

def evaluate(model, limit):
    all_ctx_sizes = [32768]
    gold_index_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    num_needles_list = [16]
    # num_needles_list = [1, 2, 4, 8, 16]
    # all are long enough in base_ctx
    base_ctx = []
    all_ctx_acc_list = []
    generate_base_ctx(model.tokenizer, base_ctx, max(all_ctx_sizes), limit)
    for ctx_size in all_ctx_sizes:
        print('Ctx size: ', ctx_size)
        trimmed_base_ctx = trim(model.tokenizer, base_ctx, ctx_size)
        final_acc_list = []
        random.seed(42)
        for needle_idx, num_needles in enumerate(num_needles_list):
            print('Num needles: ', num_needles)
            cities = []
            for _ in range(num_needles):
                city = random.choice(RANDOM_NEEDLE_CITIES)
                while city in cities:
                    city = random.choice(RANDOM_NEEDLE_CITIES)
                cities.append(city)
            magic_numbers = [random.randint(1, 50000) for _ in range(num_needles)]
            # needles[0] is the answer needle
            needles = [f"\nThe special magic {city} number is: {magic_number}.\n" for city, magic_number in zip(cities, magic_numbers)]
            retrieval_question = f"What is the special magic {cities[0]} number?"
            answer = str(magic_numbers[0])
            
            distractor_ratios = [[random.random() for _ in range(num_needles - 1)] for _ in range(len(trimmed_base_ctx))]
                
            acc_list = []
            for ratio in gold_index_ratios:
                needle_ctx = insert_needle(model.tokenizer, trimmed_base_ctx, needles, ctx_size, ratio, distractor_ratios)
                prompt_needle_ctx = insert_prompt(needle_ctx, retrieval_question, cities[0])
                acc = model_generation(model, ratio, prompt_needle_ctx, answer, limit)
                acc_list.append(round(acc, 3))
            print(f"All Acc for {ctx_size} ctx size:")
            print(acc_list)
            print(f"Average Acc for {ctx_size} ctx size: {sum(acc_list) / len(acc_list)}")
            final_acc_list.append(sum(acc_list) / len(acc_list))
        all_ctx_acc_list.append(final_acc_list)
        print(final_acc_list)
    print(all_ctx_acc_list)

def main():
    config = Config.from_args()
    llm = LLM(config)
    evaluate(llm, 10)


if __name__ == "__main__":
    main()
