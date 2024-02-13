# WIP: Time x Value Trade-off

Due to the vast size of data utilised by LLMs, an important consideration is the ability to process this data efficiently. LangChain provides 4 chunking strategies for question answering as standard; Stuffing, Map_Reduce, Refine, and Map_Rerank. During this analysis I will compare these various methods in time and accuracy as tokens scale linearly (due to limited funding). The accuracy will be tested by sample questions and evaluations also by an LLM.

## Introduction
Let's first start with a brief explanation of the 4 methods in question with some pros and cons.

### Stuffing

Stuffing involves directly adding the input content, such as documents or prompts, into the model's input prompt without any alterations. This approach may work for shorter inputs but becomes problematic when dealing with a significant number of tokens, as it can quickly reach the token limit. Despite its simplicity, it's not a scalable solution.

#### Pros
- **Simplicity**: The method is straightforward and requires minimal preprocessing of data.
- **Direct Use of Content**: Maintains the original context and details of the input without loss.

#### Cons
- **Token Limitation**: Quickly reaches the model's token limit, restricting the amount of data that can be processed.
- **Scalability Issues**: Not suitable for large datasets or complex inputs due to the token limit constraint.

### Map_Reduce

The MapReduce strategy aims to handle longer inputs by breaking them into smaller chunks (documents in this case), processing them in parallel, and then combining the outputs to create a final summary. This approach involves the following steps:

- **Map Step**: Each document is transformed into a prompt and context for the model. These prompts are sent to the LLM in parallel, utilizing the model's ability to process multiple requests simultaneously.
- **Reduce Step**: The individual summaries generated in the Map Step are combined to create a comprehensive summary that summarizes all the input documents. This process uses a reduction function to merge the summaries.

While MapReduce optimizes performance and parallel processing, it might lead to higher API call costs and potential loss of context during the summarization process.

#### Pros
- **Parallel Processing**: Enhances performance by processing multiple documents simultaneously.
- **Scalability**: Can handle larger datasets by breaking them into manageable chunks.

#### Cons
- **Higher API Costs**: Parallel processing may increase the number of API calls, leading to higher costs.
- **Potential Loss of Context**: Summarizing documents individually before combining them might lead to a loss of overall context or coherence.

### Refine

The Refine strategy draws inspiration from the functional programming concept of "foldl." It involves iteratively summarizing and refining the input by combining each successive summary with the next document, resulting in a gradually refined output. The process entails:

- **Foldl Analog**: Instead of numerical multiplication, a binary function is used to combine documents and their summaries. The initial value is an empty document or an initial summary, and the function accumulates the content.
- **Refine Chain**: The Lang Chain framework simplifies this process by automating the iterative refinement. It manages the accumulation and refinement of summaries, reducing the need for manual control.

The Refine strategy is elegant and efficient, producing a refined summary through successive iterations. It showcases the capability of Lang Chain in managing complex operations.

#### Pros
- **Iterative Refinement**: Produces a progressively refined output, improving clarity and coherence with each step.
- **Efficiency**: Reduces the need for manual intervention by automating the refinement process.

#### Cons
- **Complexity**: The process can be complex to implement, requiring a good understanding of functional programming concepts.
- **Potential for Over-Refinement**: There is a risk of losing important details through excessive refinement.

### Map_Rerank

Map_Rerank is a sophisticated strategy designed to enhance the relevance and accuracy of responses by leveraging a two-step process: initial ranking followed by reranking based on more refined criteria. This approach is particularly useful in scenarios where the initial set of results or responses needs to be optimized for quality, relevance, or other specific metrics. The process involves:

- **Initial Map Step**: In the first phase, a broad query is used to generate a wide range of responses or documents. This step is analogous to casting a wide net to ensure no potential candidate is missed. The focus here is on quantity, ensuring a comprehensive set of items for further analysis.
- **Rerank Step**: Following the initial mapping, the rerank step applies more sophisticated or specific criteria to reorder the initial set of items. This might involve additional processing, such as deeper language understanding, contextual analysis, or other forms of evaluation tailored to the specific needs of the task. The goal is to prioritize the most relevant, accurate, or otherwise valuable items from the initial set.
- **Evaluation and Optimization**: The rerank phase often includes mechanisms for evaluating the effectiveness of the reranking criteria, allowing for iterative refinement. Techniques such as A/B testing, feedback loops, or machine learning models may be employed to continuously improve the reranking process.

Map_Rerank excels in environments where the initial retrieval might produce a large set of potential matches, but the quality or relevance of those matches varies significantly. By applying a two-tiered approach, it ensures that users or downstream processes receive the most pertinent information, enhancing both user experience and operational efficiency. This strategy is commonly used in search engines, recommendation systems, and content filtering applications, where precision and relevance are paramount.

#### Pros
- **Enhanced Relevance and Accuracy**: Prioritizes the most relevant and accurate information by refining the initial set of results.
- **Iterative Optimization**: Allows for continuous improvement of the reranking process through feedback and evaluation techniques.

#### Cons
- **Increased Processing Time**: The two-step process can be more time-consuming than direct retrieval methods.
- **Complexity in Implementation**: Requires sophisticated algorithms and processing capabilities to effectively rerank and evaluate results.

## Design

[Actual]
1. Load vector db or create it from csv if it doesn't exist
2. Configure LLM
3. Create queries of increasing difficulty
4. Run queries against LLM recording
    - time
    - tokens used
    - accuracy of output (correctness) #TODO: evaluate

 [Projected]
1. Create a basic chain in LangChain
2. Embed data into a local vector db
3. Create a basic prompt template 
4. Use LLM to generate questions and answers to evaluate
5. Run evaluation against each QA
6. Record results and analyse

## Results

| Processing Type | 100 tokens | 1000 tokens | 10000 tokens |
| --------------- | ---------- | ----------- | ------------ |
| Stuff           ||||
| Map_Reduce      ||||
| Refine          ||||
| Map_Rerank      ||||
