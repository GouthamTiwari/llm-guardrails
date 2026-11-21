"""Gate an agent's tool calls before they run.

Run:  uv run python examples/agent_tool_gate.py      (no model calls at all)

Tools the policy lists as risky (in banking_tax: payments.transfer, db.execute) are always
blocked here: they need a human approval step this library cannot provide. Every other
call has its arguments scanned for denied actions (for example DROP TABLE).
"""

from llm_guardrails.adapters import check_tool_call
from llm_guardrails.policy_config import load_policy

POLICY = load_policy("banking_tax")

TOOLS = {
    "accounts.balance": lambda account: f"balance of {account}: Rs 12,500",
    "payments.transfer": lambda to, amount: f"sent Rs {amount} to {to}",
    "reports.query": lambda sql: f"ran {sql}",
}


def run_tool(name: str, **args) -> str:
    verdict = check_tool_call(name, args, POLICY)
    if verdict.action.value != "allow":
        return f"BLOCKED  {name} {verdict.reasons}"
    return f"ran      {name} -> {TOOLS[name](**args)}"


def main() -> None:
    print(run_tool("accounts.balance", account="XXXX1234"))
    print(run_tool("payments.transfer", to="XXXX9876", amount=50000))
    print(run_tool("reports.query", sql="DROP TABLE customers"))
    print(run_tool("reports.query", sql="SELECT branch, count(*) FROM loans GROUP BY branch"))


if __name__ == "__main__":
    main()
