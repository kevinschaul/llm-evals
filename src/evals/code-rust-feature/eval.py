"""
Code Rust feature implementation evaluation using inspect_ai
"""
import subprocess
import tempfile
import os
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.solver import generate, system_message, Solver
from inspect_ai.tool import Tool, ToolError, tool

@tool
def git_repo_setup():
    """Set up the git repository for code evaluation"""
    async def setup(state):
        # Clone the repository
        try:
            result = subprocess.run(
                ["git", "clone", "https://github.com/kevinschaul/jump-start-tools.git", "."],
                cwd=state.working_directory,
                capture_output=True,
                text=True,
                check=True
            )
            return f"Repository cloned successfully"
        except subprocess.CalledProcessError as e:
            raise ToolError(f"Failed to clone repository: {e}")

    return setup

@tool
def get_git_diff():
    """Get git diff to see changes made"""
    async def diff(state):
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=state.working_directory,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Git diff failed: {e}"

    return diff

def create_code_dataset() -> MemoryDataset:
    """Create a simple dataset for the code generation task"""
    samples = [
        Sample(
            input='Implement the mode=git feature for the "use" subcommand.',
            target="git_feature_implemented"  # We'll check for specific implementation markers
        )
    ]
    return MemoryDataset(samples)

@scorer(metrics=[mean()])
def code_implementation_scorer():
    """Score code implementation based on git diff and functionality"""
    async def score(state, target):
        try:
            # Get the git diff to see what was implemented
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=state.working_directory or ".",
                capture_output=True,
                text=True
            )

            diff_output = diff_result.stdout

            # Check for key implementation indicators
            implementation_score = 0.0
            explanations = []

            # Look for git-related changes
            if "git" in diff_output.lower():
                implementation_score += 0.3
                explanations.append("Git-related changes detected")

            # Look for mode parameter handling
            if "mode" in diff_output.lower():
                implementation_score += 0.3
                explanations.append("Mode parameter handling detected")

            # Look for use subcommand modifications
            if "use" in diff_output.lower():
                implementation_score += 0.2
                explanations.append("Use subcommand modifications detected")

            # Check if there are any actual code changes
            if len(diff_output.strip()) > 0:
                implementation_score += 0.2
                explanations.append("Code changes present")

            return Score(
                value=implementation_score,
                explanation=f"Implementation indicators: {'; '.join(explanations)}\nDiff length: {len(diff_output)} chars"
            )
        except Exception as e:
            return Score(
                value=0.0,
                explanation=f"Failed to evaluate implementation: {e}"
            )

    return score

class CodeGenerationSolver(Solver):
    """Custom solver for code generation tasks"""

    async def __call__(self, state, generate):
        # Set up the repository first
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/kevinschaul/jump-start-tools.git", "."],
                cwd=state.working_directory or ".",
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            # Repository might already exist, continue
            pass

        # Generate the code
        return await generate(state)

@task
def code_rust_feature():
    """Code Rust feature implementation task"""

    prompt = """You are a skilled Rust developer. Implement the mode=git feature for the "use" subcommand in this codebase.

Look at the existing code structure and implement the feature according to the patterns you see. Focus on:
1. Adding git mode support to the use subcommand
2. Following existing code patterns and conventions
3. Making the implementation robust and well-integrated

The task is: {input}

Please implement this feature by modifying the appropriate files."""

    return Task(
        dataset=create_code_dataset(),
        solver=[
            CodeGenerationSolver(),
            system_message(prompt),
            generate(cache=True)
        ],
        scorer=code_implementation_scorer(),
        config=GenerateConfig(temperature=0.1),
        sandbox="docker"  # Use sandboxing for code execution
    )