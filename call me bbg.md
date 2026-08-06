

<!-- Start of picture text -->
y<br><!-- End of picture text -->

Introduction to function calling in LLMs 

call me maybe 

## ● **Phase outcomes:** 

- Develop both general-purpose and domain-specific prompting skills. 

- Boost your productivity with effective use of AI tools. 

- Continue strengthening computational thinking, problem-solving, adaptability, and collaboration. 

## ● **Comments and examples:** 

- You’ll regularly encounter situations — exams, evaluations, and more — where you must demonstrate real understanding. Be prepared, keep building both your technical and interpersonal skills. 

- Explaining your reasoning and debating with peers often reveals gaps in your understanding. Make peer learning a priority. 

- AI tools often lack your specific context and tend to provide generic responses. Your peers, who share your environment, can offer more relevant and accurate insights. 

- Where AI tends to generate the most likely answer, your peers can provide alternative perspectives and valuable nuance. Rely on them as a quality checkpoint. 

#### ✓ **Good practice:** 

I ask AI: “How do I test a sorting function?” It gives me a few ideas. I try them out and review the results with a peer. We refine the approach together. 

#### ✗ **Bad practice:** 

I ask AI to write a whole function, copy-paste it into my project. During peer— evaluation, I can’t explain what it does or why. I lose credibility and I fail my project. 

#### ✓ **Good practice:** 

I use AI to help design a parser. Then I walk through the logic with a peer. We catch — two bugs and rewrite it together better, cleaner, and fully understood. 

#### ✗ **Bad practice:** 

I let Copilot generate my code for a key part of my project. It compiles, but I can’t explain how it handles pipes. During the evaluation, I fail to justify and I fail my project. 

4 

# **Chapter III** 

# **Introduction** 

## **III.1 What is Function Calling?** 

Large Language Models (LLMs) are powerful at understanding and generating human language, but they don’t naturally produce structured, machine-executable output. Function calling bridges this gap by translating natural language requests into precise function calls with typed arguments. 

Consider this example: Natural Language to Function Call 

```
User:"Whatisthesumof40and2?"
TraditionalLLM:"Thesumof40and2is42."
FunctionCallingSystem:
{
"function":"add_numbers",
"arguments":{"a":40,"b":2}
}
```

The function calling system doesn’t answer the question directly. Instead, it provides the **tools** to solve it: the right function name and the correct arguments with proper types. 

5 



<!-- Start of picture text -->
ee<br>eee<br><!-- End of picture text -->

# **Chapter V** 

# **Mandatory part** 

## **V.1 Summary** 

In this project, you will create a function calling tool that translates natural language prompts into structured function calls. Given a question like "What is the sum of 40 and 2?", your solution should not return 42, but instead provide: 

- The function name: `fn_add_numbers` 

- The arguments: `{"a": 40, "b": 2}` 

Your implementation must use **constrained decoding** to guarantee 100% valid JSON output, ensuring near-perfect reliability even with a small 0.6B parameter model. 

## **V.2 Input Files** 

Your solution will process two input files located in the `data/input/` directory: 

- `function_calling_tests.json` : contains a JSON array of natural language prompts that your system must process. 

Example: function_calling_tests.json 



<!-- Start of picture text -->
[<br>{<br>"prompt": "What is the sum of 2 and 3?"<br>},<br>{<br>"prompt": "What is the sum of 265 and 345?"<br>},<br>{<br>"prompt": "Greet shrek"<br>},<br>{<br>"prompt": "Greet john"<br>},<br>{<br>"prompt": "Reverse the string 'hello'"<br>},<br>...<br>]<br><!-- End of picture text -->

10 

Introduction to function calling in LLMs 

call me maybe 

- `functions_definition.json` : contains the available functions your system can call. Each function includes: 

   - Function name 

   - Argument names and types 

   - Return type 

   - Description 

Example: functions_definition.json 

```
[
{
"name":"fn_add_numbers",
"description":"Addtwonumberstogetherandreturntheirsum.",
"parameters":{
"a":{
"type":"number"
},
"b":{
"type":"number"
}
},
"returns":{
"type":"number"
}
},
{
"name":"fn_greet",
"description":"Generateagreetingmessageforapersonbyname.",
"parameters":{
"name":{
"type":"string"
}
},
"returns":{
"type":"string"
}
},
{
"name":"fn_reverse_string",
"description":"Reverseastringandreturnthereversedresult.",
"parameters":{
"s":{
"type":"string"
}
},
"returns":{
"type":"string"
}
},
...
]
```



```
Theseexamplesestablishtheexpectedcomplexitylevel.However,
yoursolutionwillbetestedwithdifferentpromptsandfunction
sets.YoumustimplementproperJSONerrorhandlingforinputfiles,
astheymaycontaininvalidJSONorbemissingentirely.
```

11 

rN <mark>O</mark> 

Introduction to function calling in LLMs 

call me maybe 

## **V.4 Output File Format** 

Your program will produce a single JSON file: `data/output/function_calling_results.json` . For each prompt, add a JSON object to this file. Each object in the array must contain exactly the following keys: 

- `prompt` (string): The original natural-language request 

- `name` (string): The name of the function to call 

- `parameters` (object): All required arguments with the correct types 

### **V.4.1 Example Output** 

```
[
{
"prompt":"Whatisthesumof2and
"name":"fn_add_numbers",
"parameters":{"a":2.0,"b":3.0}
},
{
"prompt":"Reversethestring'hello'",
"name":"fn_reverse_string",
"parameters":{"s":"hello"}
}
]
```

```
"prompt":"Whatisthesumof2and3?",
"name":"fn_add_numbers",
"parameters":{"a":2.0,"b":3.0}
```

### **V.4.2 Validation Rules** 

- The file must be valid JSON (no trailing commas, no comments) 

- Keys and types must match the schema in `functions_definition.json` exactly 

- No extra keys or prose are allowed anywhere in the output 

- All required arguments must be present 

- Argument types must match the function definition (number, string, boolean, etc.) 



```
Thegiveninputfilesmaychangeduringthepeerreview.Donot
hardcodesolutionsbasedontheprovidedexamples.
```

14 







# **Chapter VIII** 

# **Submission and review peer** 

Submit your assignment in your `Git` repository as usual. Only the work inside your repository will be reviewed during the defense. Don’t hesitate to double-check the names of your files to ensure they are correct. 

Your repository must contain: 

- `src/` directory with your implementation 

- `pyproject.toml` and `uv.lock` for dependency management 

- `llm_sdk/` directory (copied from the provided package) 

- `data/input/` directory with test files (for demonstration) 

- `README.md` with comprehensive documentation 

- Any additional files needed to run your solution 



```
Donotincludetheoutput/directoryinyourrepository.Itwillbe
generatedduringthepeerreview.
```

During the evaluation, a brief **modification of the project** may occasionally be requested. This could involve a minor behaviour change, a few lines of code to write or rewrite, or an easy-to-add feature. 

While this step may **not be applicable to every project** , you must be prepared for it if it is mentioned in the evaluation guidelines. 

This step is meant to verify your actual understanding of a specific part of the project. The modification can be performed in any development environment you choose (e.g., your usual setup), and it should be feasible within a few minutes — unless a specific time frame is defined as part of the evaluation. 

19 

