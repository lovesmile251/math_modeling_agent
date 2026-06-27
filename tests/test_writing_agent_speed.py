from __future__ import annotations

from agents.base import A_PAPER, K_RESULT_ANALYSIS, K_WRITING_MODE, WorkflowState
from agents.writing_agent import WritingAgent


class MockLLM:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, instructions: str, user_input: str) -> str:
        self.calls.append((instructions, user_input))
        return "\n".join(
            [
                "# 快速论文草稿",
                "## 摘要",
                "本文基于真实结果数据完成建模分析。",
                "## 结果分析",
                "模型运行结果见生成表格。",
            ]
        )


def test_fast_writing_mode_uses_single_llm_call(monkeypatch, temp_workspace):
    monkeypatch.setenv("MMA_LLM_FAST_MODE", "1")
    llm = MockLLM()
    state = WorkflowState(
        problem_text="建立需求预测模型并撰写论文。",
        data_files=[],
        workspace=temp_workspace,
        llm=llm,
    )
    state.notes[K_RESULT_ANALYSIS] = "趋势预测模型已输出结果表。"

    result = WritingAgent().run(state)

    assert result.notes[K_WRITING_MODE] == "single_llm_fast"
    assert result.notes["writing_agent_fast_mode"] == "true"
    assert len(llm.calls) == 1
    assert result.artifacts[A_PAPER].exists()
