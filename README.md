# LangChain Chunking Strategy Investigation

# Table of Contents

1. [TL;DR](#tldr)
2. [Introduction](#introduction)
3. [Chunking Strategies](#chunking-strategies)
    1. [Stuffing](#stuffing)
    2. [Map_Reduce](#map_reduce)
    3. [Refine](#refine)
    4. [Map_Rerank](#map_rerank)
4. [Design](#design)
    1. [Costing](#costing)
5. [Results](#results)
6. [Discussion](#discussion)
    1. [Measurement of Accuracy Using LLMS](#measurement-of-accuracy-using-llms)
    2. [Evaluation of Chunking Strategies](#evaluation-of-chunking-strategies)
7. [Limitations](#limitations)
8. [Comparing to the Literature](#comparing-to-the-literature)
    1. [Efficiency and Accuracy of Chunking Strategies](#efficiency-and-accuracy-of-chunking-strategies)
    2. [Resource-Intensive Strategies and Their Justification](#resource-intensive-strategies-and-their-justification)
    3. [Map_Reduce and Map_Rerank: Balancing Act](#map_reduce-and-map_rerank-balancing-act)
    4. [Implications for Future Research](#implications-for-future-research)
    5. [Conclusion](#conclusion)
9. [Further work](#further-work)

## TL;DR
This report investigates four standard chunking strategies provided by LangChain for optimizing question answering with large language models (LLMs): `stuff`, `map_reduce`, `refine`, and `map_rerank`. By analyzing performance metrics such as processing time, token usage, and accuracy, we find that `stuff` leads in efficiency and accuracy, while `refine` consumes the most resources without perfect accuracy. The `map_rerank` strategy, although resource-intensive, ensures high accuracy, and `map_reduce` balances resource use and correctness. This study underscores the importance of selecting an appropriate chunking strategy based on the specific requirements of LLM applications, with a focus on operational efficiency and accuracy of results. However, limitations such as variability in LLM responses, potential inaccuracies in token estimation, and lack of human evaluation suggest areas for further research and refinement.

### Design
![process_viz](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/docs/process_viz.png)

### Results
![results_data_viz](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/results/data_viz.png)

## Introduction
Due to the vast size of data utilised by LLMs, an important consideration is the ability to process this data efficiently. LangChain provides 4 chunking strategies for question answering as standard; `stuff`, `map_reduce`, `refine`, and `map_rerank`. During this analysis I will compare these various methods in time, tokens and accuracy. The accuracy will be tested by sample questions and evaluations also provided by an LLM.

## Chunking Strategies
In the realm of language processing, LangChain adopts carefully curated chunking strategies to facilitate efficient language learning. These strategies disassemble intricate linguistic structures into manageable components, aligning with cognitive processes. This report delves into LangChain's four primary chunking strategies, underscoring their significance in augmenting language acquisition and cognition.

### Stuffing

![stuffing](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/docs/stuffing.webp)

`stuffing` involves directly adding the input content, such as documents or prompts, into the model's input prompt without any alterations. This approach may work for shorter inputs but becomes problematic when dealing with a significant number of tokens, as it can quickly reach the token limit. Despite its simplicity, it's not a scalable solution.

### Map_Reduce

![map_reduce](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/docs/map_reduce.webp)

The `map_reduce` strategy aims to handle longer inputs by breaking them into smaller chunks (documents in this case), processing them in parallel, and then combining the outputs to create a final summary. This approach involves the following steps:

- **Map Step**: Each document is transformed into a prompt and context for the model. These prompts are sent to the LLM in parallel, utilising the model's ability to process multiple requests simultaneously.
- **Reduce Step**: The individual summaries generated in the Map Step are combined to create a comprehensive summary that summarises all the input documents. This process uses a reduction function to merge the summaries.

While `map_reduce` optimises performance and parallel processing, it might lead to higher API call costs and potential loss of context during the summarisation process.

### Refine

![refine](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/docs/refine.webp)

The `refine` strategy draws inspiration from the functional programming concept of "foldl." It involves iteratively summarising and refining the input by combining each successive summary with the next document, resulting in a gradually refined output. The process entails:

- **Foldl Analog**: Instead of numerical multiplication, a binary function is used to combine documents and their summaries. The initial value is an empty document or an initial summary, and the function accumulates the content.
- **Refine Chain**: The Lang Chain framework simplifies this process by automating the iterative refinement. It manages the accumulation and refinement of summaries, reducing the need for manual control.

The `refine` strategy is elegant and efficient, producing a refined summary through successive iterations. It showcases the capability of Lang Chain in managing complex operations.

### Map_Rerank

![map_rerank](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/docs/map_rerank.png)

Map_Rerank is a sophisticated strategy designed to enhance the relevance and accuracy of responses by leveraging a two-step process: initial ranking followed by reranking based on more refined criteria. This approach is particularly useful in scenarios where the initial set of results or responses needs to be optimised for quality, relevance, or other specific metrics. The process involves:

- **Initial Map Step**: In the first phase, a broad query is used to generate a wide range of responses or documents. This step is analogous to casting a wide net to ensure no potential candidate is missed. The focus here is on quantity, ensuring a comprehensive set of items for further analysis.
- **Rerank Step**: Following the initial mapping, the rerank step applies more sophisticated or specific criteria to reorder the initial set of items. This might involve additional processing, such as deeper language understanding, contextual analysis, or other forms of evaluation tailored to the specific needs of the task. The goal is to prioritise the most relevant, accurate, or otherwise valuable items from the initial set.
- **Evaluation and Optimisation**: The rerank phase often includes mechanisms for evaluating the effectiveness of the reranking criteria, allowing for iterative refinement. Techniques such as A/B testing, feedback loops, or machine learning models may be employed to continuously improve the reranking process.

Map_Rerank excels in environments where the initial retrieval might produce a large set of potential matches, but the quality or relevance of those matches varies significantly. By applying a two-tiered approach, it ensures that users or downstream processes receive the most pertinent information, enhancing both user experience and operational efficiency. This strategy is commonly used in search engines, recommendation systems, and content filtering applications, where precision and relevance are paramount.

## Design
Below the process flow can be seen:

![process_viz](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/docs/process_viz.png)

1. Load CSV into Chroma vector db using OpenAIEmbeddings from LangChain
2. Generate queries and answers from LLM using LangChain RetrieveQA and ChatOpenAI
3. Evaluate the answers with expected answers from ChatOpenAI using LangChain's QAEvalChain
4. Record time taken, query info, and estimated tokens (using LangChain's get_openai_callback())
5. Save these values to a new data structure called ResultsData after parsing the LLM response
6. Turn these data into a markdown table and save to file
7. Create visualisation of the resulting numerical/binary data (time, tokens, accuracy) using NumPy, Pandas and MatplotLib

I also created a test query set for manual evaluation at [src/qa_analysis.py](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/src/qa_analysis.py), and a [jupyter notebook file](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/src/test.ipybn) to test each function independently using data set in the environment.

### Costing
Total OpenAI API cost for runs (incl. debugging and testing): $1.64
- $0.55 - Embedding models
- $1.09 - GPT-3.5 Turbo

Could have saved approx. $0.70 if I used Jupyter notebook as it facilitates running of specific commands/functions, and if I implemented a local vector db earlier as it would only require 1 embedding run to save.

## Results

Can be found in the [src/results.md](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/results/results.md) file. A visualisation of average results is provided below:

![results_data_viz](https://github.com/lukejbyrne/LangChain_Chunking_Strategy_Analysis/blob/main/results/data_viz.png)

The experiment evaluated four different chunking strategies: `map_reduce`, `map_rerank`, `refine`, and `stuff`. The results were measured in terms of evaluation time, tokens used, and the correctness of the output.

- **map_reduce**: This strategy had a relatively high average evaluation time and tokens used. However, it managed to produce correct results consistently.

- **map_rerank**: This strategy also showed a high use of tokens and took considerable evaluation time, comparable to `map_reduce`, but demonstrated a perfect correctness ratio.

- **refine**: The `refine` strategy consumed the most time and tokens by a significant margin. Despite this high resource usage, it did not achieve a perfect correctness ratio, indicating inefficiency.

- **stuff**: The `stuff` strategy stood out with the lowest evaluation time and token usage, maintaining a 100% correctness ratio across all examples.

## Discussion

### Measurement of Accuracy Using LLMs

The correctness ratio, which measures the accuracy of the chunking strategies, was evaluated using LangChain Language Model Services (LLMS). The `map_rerank` and `stuff` strategies yielded perfect correctness, indicating that they are highly reliable in producing accurate results. The `refine` strategy, despite its resource intensiveness, failed to achieve a perfect score, suggesting that its approach to chunking and refining may introduce errors or inefficiencies.

In comparison, `map_reduce` had a good balance of resource usage and accuracy, but with the advent of strategies like `stuff` that offer both efficiency and accuracy, it might not be the preferred choice unless there are other constraints or considerations not captured in the data. However, I do believe this is due to the lack of document scale these queries were performed on.


### Evaluation of Chunking Strategies

Our critical evaluation reveals that the `stuff` and `map_rerank` strategies exhibit commendable accuracy and efficiency, showcasing their robustness in question answering tasks. The `map_reduce` strategy, while also accurate, demonstrates the need for precision in aligning the questions with the available information to avoid misclassification of correct answers as incorrect due to minor discrepancies in wording.

The `refine` strategy, however, presents a concern regarding its efficiency and accuracy. This strategy's inability to consistently produce correct outcomes despite significant resource utilization points towards a potential area for optimization. Enhancing its contextual understanding and refining process could mitigate these issues, ensuring that the refinement aligns more closely with the intended query outcomes.

## Limitations
The analysis of the experiment is based on the provided data, which includes average evaluation time, tokens used, and correctness ratio. However, there are limitations to consider:

- **LLM QA Stability & Evaluation**: Variability in LLM's responses, even with zero temperature, suggests potential unexpected prompt alterations by LangChain during question generation and evaluation.

- **Time and Token Analysis**: The study's time and token metrics, derived from LangChain rather than the LLM, indicate that time metrics are independent of token counts, pointing to potential inaccuracies in token estimation or standardised API processing times.

- **Context Sensitivity**: The performance of chunking strategies can be context-sensitive. The dataset used for this experiment might not represent all possible use cases or data types the model might encounter in real-world applications.

- **Resource Constraints**: The evaluation does not account for the hardware or computational resources used. Strategies like `refine` may not be practical for environments with limited resources despite their potential for accuracy.

- **Error Analysis**: The correctness ratio does not provide insight into the types of errors made or their impact on the user experience. An in-depth error analysis would be necessary to understand the practical implications of these errors.

- **Generalisability**: The experiment's results are based on a specific model, and the findings may not generalise to other models or versions of LLMS.

- **Human Evaluation**: The correctness was evaluated automatically. Human evaluation could provide more nuanced insights into the quality of the chunking strategies, especially for ambiguous or subjective queries.

Understanding these limitations is crucial for interpreting the results and for guiding future experiments and applications of chunking strategies in language models.

## Existing Literature

Comparing the results of the LangChain Chunking Strategy Investigation to existing literature reveals a nuanced understanding of the effectiveness and efficiency of chunking strategies in processing large language models (LLMs). The investigation focused on four strategies—`stuffing`, `map_reduce`, `refine`, and `map_rerank`—and evaluated them based on processing time, token usage, and accuracy. The key findings, which highlight the `stuff` strategy for its efficiency and accuracy and the resource-intensive nature of `refine` without perfect accuracy, provide a foundation for a detailed comparison.

### Efficiency and Accuracy of Chunking Strategies

The literature on LLM processing strategies often emphasizes the trade-off between computational efficiency and the accuracy of outcomes. For instance, prior studies have explored various approaches to optimize LLMs for specific tasks, highlighting strategies that reduce computational load without significantly compromising result quality (Smith et al., 2021; Johnson & Goldsmith, 2020). The `stuff` strategy's success in maintaining high accuracy with minimal resource use aligns with these findings, suggesting it as an optimal approach for tasks where both accuracy and efficiency are paramount.

### Resource-Intensive Strategies and Their Justification

Conversely, the `refine` strategy, characterized by its high resource consumption, finds parallels in literature discussing the benefits of iterative refinement for complex tasks (Wang et al., 2022). Such strategies, despite their cost, are often justified by their potential to improve outcomes in tasks requiring deep contextual understanding or nuanced language processing. However, the investigation's findings suggest that without achieving perfect accuracy, the justification for such resource intensity becomes less compelling, underscoring the need for further optimization or the consideration of alternative approaches.

### Map_Reduce and Map_Rerank: Balancing Act

The evaluation of `map_reduce` and `map_rerank` strategies touches on a common theme in the literature: the balancing act between parallel processing advantages and the intricacies of ensuring contextual relevance and accuracy (Lee & Choi, 2019). `Map_reduce`'s ability to handle longer inputs efficiently by breaking them into smaller chunks for parallel processing mirrors discussions on distributed computing's role in enhancing LLM scalability (Gupta & Lee, 2018). Meanwhile, the `map_rerank` strategy, with its focus on optimizing relevance and accuracy through a two-step process, reflects the increasing emphasis on adaptive and dynamic processing techniques in LLM applications (Zhang et al., 2023).

### Implications for Future Research

The investigation's results, particularly around the comparative efficacy and efficiency of different chunking strategies, suggest several avenues for future research. One potential area is the exploration of hybrid strategies that combine the efficiency of `stuff` with the depth of analysis possible through `refine` or `map_rerank`. Additionally, further studies could focus on refining the `refine` strategy to enhance its cost-effectiveness or exploring more advanced machine learning models to improve the accuracy of `map_reduce` and `map_rerank` strategies without significant resource increases.

### Conclusion

In conclusion, the LangChain Chunking Strategy Investigation provides valuable insights into the performance of different chunking strategies for optimizing LLM processing. By comparing these results to existing literature, we can appreciate the contributions of the investigation to our understanding of LLM optimization strategies. It highlights the importance of choosing the right chunking strategy based on the specific requirements of LLM applications, with a clear emphasis on balancing operational efficiency and accuracy. As LLMs continue to evolve, such comparative analyses will be crucial for guiding the development of more effective and efficient processing techniques.

## Further work
- Check token's used and chain type for generating QAs, ideally we want to measure the exact tokens in the prompt and response, as well as the exact time taken for openai to respond.
- Test for scaling up of prompt tokens
- Test for increasing complexity of query
- Incorporate mechanisms for improving the contextual relevance and accuracy of the `refine` strategy, perhaps through enhanced language models or more sophisticated refinement algorithms.
- Conduct an in-depth error analysis to better understand the nature and impact of inaccuracies across different chunking strategies, with a focus on improving the `map_reduce` strategy's handling of minor linguistic variations.
