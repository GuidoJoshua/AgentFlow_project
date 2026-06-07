import json
import os
import re
from typing import Any, Dict, List, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import NextStep, PlannedStep, QueryAnalysis
from agentflow.models.memory import Memory


class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "gpt-4o",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, fixed_base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature : float = .0, tool_selection_mode: str = "embedding"):
        if tool_selection_mode not in {"planner", "embedding"}:
            raise ValueError("tool_selection_mode must be either 'planner' or 'embedding'.")
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        self.tool_selection_mode = tool_selection_mode
        # self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        self.llm_engine_fixed = create_llm_engine(
            model_string=llm_engine_fixed_name,
            base_url=fixed_base_url,
            is_multimodal=False,
            temperature=temperature,
        )
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []

        self.verbose = verbose
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: int = 2048) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")


        print("Input data of `generate_base_response()`: ", input_data)
        # self.base_response = self.llm_engine(input_data, max_tokens=max_tokens)
        self.base_response = self.llm_engine_fixed(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)

        if self.is_multimodal:
            tool_guidance = (
                "4. Examine the available tools in the toolbox and identify the capability requirements "
                "needed from future steps. Do not make a final tool choice."
                if self.tool_selection_mode == "embedding"
                else "4. Examine the available tools in the toolbox and determine which ones might relevant "
                "and useful for addressing the query. Make sure to consider the user metadata for each tool, "
                "including limitations and potential applications (if available)."
            )
            output_guidance = (
                "3. A list of relevant tool capabilities from the toolbox, with a brief explanation of how "
                "each capability would be utilized and its potential limitations."
                if self.tool_selection_mode == "embedding"
                else "3. A list of relevant tools from the toolbox, with a brief explanation of how each tool "
                "would be utilized and its potential limitations."
            )
            query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
{tool_guidance}
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
{output_guidance}
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
                        """
        else: 
            instruction_line = (
                "2. List the necessary skills and capability requirements for the next step. Do not make a final tool choice; "
                "focus on what kind of tool behavior is needed."
                if self.tool_selection_mode == "embedding"
                else "2. List the necessary skills"
            )
            query_prompt = f"""
Task: Analyze the given query to determine necessary skills.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
{instruction_line}
3. For each skill, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills with explanations, and a section for additional considerations.

Be brief and precise with insight. 
"""


        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        print("Input data of `analyze_query()`: ", input_data)

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

    def _normalize_tool_name(self, tool_name: str) -> str:
        def to_canonical(name: str) -> str:
            parts = re.split("[ _]+", name)
            return "_".join(part.lower() for part in parts)

        normalized_input = to_canonical(tool_name)

        for tool in self.available_tools:
            if to_canonical(tool) == normalized_input:
                return tool

        return f"No matched tool given: {tool_name}"

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        stripped = text.lstrip()
        return stripped.startswith("{") or stripped.startswith("[")

    def _coerce_planned_response(self, response: Any) -> Any:
        if isinstance(response, (PlannedStep, NextStep)):
            return response

        if isinstance(response, dict):
            try:
                if "tool_name" in response:
                    return NextStep(**response)
                return PlannedStep(**response)
            except Exception:
                return response

        if isinstance(response, str) and self._looks_like_json(response):
            try:
                response_dict = json.loads(response)
                if "tool_name" in response_dict:
                    return NextStep(**response_dict)
                return PlannedStep(**response_dict)
            except Exception:
                return response

        return response

    def extract_context_and_subgoal(self, response: Any) -> Tuple[str, str]:
        try:
            response = self._coerce_planned_response(response)
            if isinstance(response, (PlannedStep, NextStep)):
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                return context, sub_goal

            if isinstance(response, dict):
                context = response.get("context")
                sub_goal = response.get("sub_goal")
                return (
                    context.strip() if isinstance(context, str) else None,
                    sub_goal.strip() if isinstance(sub_goal, str) else None,
                )

            if isinstance(response, str):
                text = response.replace("**", "")
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)(?:Tool Name:|\Z)"
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    context, sub_goal = matches[-1]
                    return context.strip(), sub_goal.strip()

            return None, None
        except Exception as e:
            print(f"Error extracting context and sub-goal: {str(e)}")
            return None, None

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:
        try:
            response = self._coerce_planned_response(response)
            context, sub_goal = self.extract_context_and_subgoal(response)

            if isinstance(response, NextStep):
                tool_name = self._normalize_tool_name(response.tool_name.strip())
                return context, sub_goal, tool_name

            if isinstance(response, PlannedStep):
                return context, sub_goal, None

            if isinstance(response, dict):
                tool_name = response.get("tool_name")
                normalized_tool_name = (
                    self._normalize_tool_name(tool_name.strip())
                    if isinstance(tool_name, str) and tool_name.strip()
                    else None
                )
                return context, sub_goal, normalized_tool_name

            if isinstance(response, str):
                text = response.replace("**", "")
                tool_match = re.findall(
                    r"Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)",
                    text,
                    re.DOTALL,
                )
                tool_name = self._normalize_tool_name(tool_match[-1].strip()) if tool_match else None
                return context, sub_goal, tool_name

            return context, sub_goal, None
        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None

    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, json_data: Any = None) -> Any:
        if self.is_multimodal and self.tool_selection_mode == "planner":
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
- Key objectives from the query analysis
- Capabilities of available tools
- Logical progression of problem-solving
- Outcomes from previous steps
- Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>

Where:
- <context> MUST include ALL necessary information for the tool to function, structured as follows:
* Relevant data from previous steps
* File names or paths created or used in previous steps (list EACH ONE individually)
* Variable names and their values from previous steps' results
* Any other context-specific information required by the tool
- <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
- <tool_name> MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {self.available_tools}.
- Avoid redundancy by considering previous steps and building on prior results.
- Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
- Include NO content after these three sections.

Example (do not copy, use only as reference):
Justification: [Your detailed explanation here]
Context: Image path: "example/image.jpg", Previous detection results: [list of objects]
Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
Tool Name: Object_Detector_Tool

Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
                        """
        elif self.is_multimodal:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis and previous steps taken.

Tool selection will be handled separately by an embedding-based selector after you produce the plan for this step.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the query, image, prior reasoning, previous steps, and tool metadata.
2. Decide the single most useful next objective that can be completed by exactly one available tool.
3. Do NOT choose a tool and do NOT include a Tool Name section.
4. Keep the sub-goal tool-agnostic: describe the required action or information, not the tool identity.
5. Provide complete execution context, including relevant prior results, file paths, URLs, variable names, and values needed by the downstream tool.

Response Format:
1. Justification: Explain why this next step is the best use of the remaining budget.
2. Context: Provide all prerequisite information needed by the downstream tool.
3. Sub-Goal: State the exact objective for the downstream tool.

Rules:
- The sub-goal must be specific, achievable in one step, and directly tied to the query.
- The sub-goal should be written so that a downstream embedding-based selector can match it to the best tool.
- Do not include any Tool Name or final tool decision in the response.
- The final response must end with the Context and Sub-Goal sections in that order. No additional text should follow.
"""
        else:
            if self.tool_selection_mode == "planner":
                prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the query using available tools and previous context.

Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the current objective, the history of executed steps, and the capabilities of the available tools.
2. Select the single most appropriate tool for the next action.
3. Consider the specificity of the task (e.g., calculation vs. information retrieval).
4. Consider the source of required information (e.g., general knowledge, mathematical computation, a specific URL).
5. Consider the limitations of each tool as defined in the metadata.
6. Formulate a clear, concise, and achievable sub-goal that precisely defines what the selected tool should accomplish.
7. Provide all necessary context (e.g., relevant data, variable names, file paths, or URLs) so the tool can execute its task without ambiguity.


Response Format:
1. Justification: Explain why the chosen tool is optimal for the sub-goal, referencing its capabilities and the task requirements.
2. Context: Provide all prerequisite information for the tool.
3. Sub-Goal: State the exact objective for the tool.
4. Tool Name: State the exact name of the selected tool (e.g., Wikipedia Search).

Rules:
- Select only one tool per step.
- The Sub-Goal must be directly and solely achievable by the selected tool.
- The Context section must contain all information the tool needs; do not assume implicit knowledge.
- The final response must end with the Context, Sub-Goal, and Tool Name sections in that order. No additional text should follow.
                    """
            else:
                prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the query using previous context.

Tool selection will be handled separately by an embedding-based selector after you produce the plan for this step.

Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the current objective, the history of executed steps, and the capabilities of the available tools.
2. Decide on the single most valuable next objective that can be completed by exactly one available tool.
3. Do NOT choose a tool and do NOT include a Tool Name section.
4. Keep the sub-goal tool-agnostic: describe the required action or information rather than a tool identity.
5. Consider tool limitations and remaining steps so the sub-goal is realistic for one execution step.
6. Provide all relevant context, including data, file paths, URLs, prior results, and variable names needed by the downstream tool.

Response Format:
1. Justification: Explain why this next step best advances the solution.
2. Context: Provide all prerequisite information for the downstream tool.
3. Sub-Goal: State the exact objective for the downstream tool.

Rules:
- The sub-goal must be concrete, concise, and achievable by one available tool.
- The sub-goal should be specific enough for a downstream embedding-based selector to match it to the best tool.
- Do not include any Tool Name or final tool choice anywhere in the response.
- The final response must end with the Context and Sub-Goal sections in that order. No additional text should follow.
"""
            
        response_format = NextStep if self.tool_selection_mode == "planner" else PlannedStep
        next_step = self.llm_engine(prompt_generate_next_step, response_format=response_format)
        # next_step = self.llm_engine_fixed(prompt_generate_next_step, response_format=NextStep)
        if json_data is not None:
            json_data[f"action_predictor_{step_count}_prompt"] = prompt_generate_next_step
            json_data[f"action_predictor_{step_count}_response"] = str(next_step)
        return next_step


    def generate_final_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""
        else:
                prompt_generate_final_output = f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine_mm(input_data)
        final_output = self.llm_engine(input_data)
        # final_output = self.llm_engine_fixed(input_data)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""
        else:
            prompt_generate_final_output = f"""
Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {self.query_analysis}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Carefully review the original user query, the initial analysis, and the complete sequence of actions and their results.
2. Synthesize the key findings from the action history into a coherent narrative.
3. Construct a clear, step-by-step summary that explains how each action contributed to solving the query.
4. Provide a direct, precise, and standalone final answer to the original query.

Output Structure:
1. Process Summary: A clear, step-by-step breakdown of how the query was addressed. For each action, state its purpose (e.g., “To verify X”) and summarize its key result or finding in one sentence.
2. Answer: A direct and concise final answer to the query. This should be a self-contained statement that fully resolves the user’s question.

Rules:
- The response must follow the exact two-part structure above.
- The Process Summary should be informative but concise, focusing on the logical flow of the solution.
- The Answer must be placed at the very end and be clearly identifiable.
- Do not include any additional sections, explanations, or disclaimers beyond the specified structure.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)
        # final_output = self.llm_engine_mm(input_data)

        return final_output
