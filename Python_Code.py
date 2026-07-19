# ============================================================
# Sequential Enterprise AI Agent
# ============================================================
# This notebook demonstrates a simple enterprise sequential
# agent architecture using Python.
#
# Workflow:
#   1. Validation Agent
#   2. Risk Assessment Agent
#   3. Compliance Agent
#   4. Approval Agent
#   5. Execution Agent
#   6. Audit Logging Agent
#
# The workflow is STRICT and PREDEFINED.
# Each step must succeed before the next step starts.
# ============================================================

#%%
# ============================================================
# IMPORTS
# ============================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

#%%
# ============================================================
# TASK OBJECT
# ============================================================

@dataclass
class EnterpriseTask:
    """
    Represents a corporate task request.
    """

    employee_name: str
    department: str
    request_type: str
    amount: float

    approved: bool = False
    rejected: bool = False
    execution_status: str = "NOT_STARTED"

    logs: List[str] = field(default_factory=list)

    def add_log(self, message):
        """
        Add a timestamped log entry.
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"[{timestamp}] {message}"

        self.logs.append(log_message)

        print(log_message)

# ============================================================
# BASE AGENT CLASS
# ============================================================

class BaseAgent:
    """
    Base class for all enterprise agents.
    """

    def __init__(self, name):
        self.name = name

    def process(self, task):
        raise NotImplementedError

#%%
# ============================================================
# VALIDATION AGENT
# ============================================================

class ValidationAgent(BaseAgent):

    def process(self, task):

        task.add_log(f"{self.name}: Starting validation.")

        # Simple validation rules
        if task.amount <= 0:
            task.rejected = True
            task.add_log(f"{self.name}: Invalid amount.")
            return False

        if task.request_type == "":
            task.rejected = True
            task.add_log(f"{self.name}: Missing request type.")
            return False

        task.add_log(f"{self.name}: Validation successful.")

        return True

#%%
# ============================================================
# RISK ASSESSMENT AGENT
# ============================================================

class RiskAssessmentAgent(BaseAgent):

    def process(self, task):

        task.add_log(f"{self.name}: Performing risk assessment.")

        # Example enterprise risk rule
        if task.amount > 50000:

            task.rejected = True

            task.add_log(
                f"{self.name}: Request exceeds risk threshold."
            )

            return False

        task.add_log(f"{self.name}: Risk assessment passed.")

        return True

#%%
# ============================================================
# COMPLIANCE AGENT
# ============================================================

class ComplianceAgent(BaseAgent):

    def process(self, task):

        task.add_log(f"{self.name}: Running compliance checks.")

        # Example compliance rule
        restricted_departments = ["Blacklisted Division"]

        if task.department in restricted_departments:

            task.rejected = True

            task.add_log(
                f"{self.name}: Department restricted by policy."
            )

            return False

        task.add_log(f"{self.name}: Compliance checks passed.")

        return True

#%%
# ============================================================
# APPROVAL AGENT
# ============================================================

class ApprovalAgent(BaseAgent):

    def process(self, task):

        task.add_log(f"{self.name}: Manager approval started.")

        # Example approval logic
        if task.amount < 10000:

            task.approved = True

            task.add_log(f"{self.name}: Automatically approved.")

            return True

        # Manual approval simulation
        elif task.amount <= 50000:

            task.approved = True

            task.add_log(
                f"{self.name}: Approved after management review."
            )

            return True

        else:

            task.rejected = True

            task.add_log(f"{self.name}: Approval rejected.")

            return False

#%%
# ============================================================
# EXECUTION AGENT
# ============================================================

class ExecutionAgent(BaseAgent):

    def process(self, task):

        task.add_log(f"{self.name}: Starting execution.")

        if not task.approved:

            task.execution_status = "FAILED"

            task.add_log(
                f"{self.name}: Cannot execute unapproved request."
            )

            return False

        # Simulate execution
        task.execution_status = "COMPLETED"

        task.add_log(f"{self.name}: Task executed successfully.")

        return True

#%%
# ============================================================
# AUDIT AGENT
# ============================================================

class AuditAgent(BaseAgent):

    def process(self, task):

        task.add_log(f"{self.name}: Recording audit information.")

        task.add_log(
            f"{self.name}: Final status = {task.execution_status}"
        )

        return True

#%%
# ============================================================
# ENTERPRISE WORKFLOW ENGINE
# ============================================================

class SequentialEnterpriseWorkflow:
    """
    Enterprise sequential workflow engine.

    Every agent runs in a strict order.
    If one agent fails, the workflow stops immediately.
    """

    def __init__(self):

        self.agents = [

            ValidationAgent("Validation Agent"),

            RiskAssessmentAgent("Risk Assessment Agent"),

            ComplianceAgent("Compliance Agent"),

            ApprovalAgent("Approval Agent"),

            ExecutionAgent("Execution Agent"),

            AuditAgent("Audit Agent")
        ]

    def run(self, task):

        task.add_log("================================================")
        task.add_log("ENTERPRISE WORKFLOW STARTED")
        task.add_log("================================================")

        for agent in self.agents:

            success = agent.process(task)

            if not success:

                task.add_log(
                    f"WORKFLOW STOPPED at {agent.name}"
                )

                task.add_log(
                    "================================================"
                )

                return False

        task.add_log("WORKFLOW COMPLETED SUCCESSFULLY")

        task.add_log("================================================")

        return True

#%%
# ============================================================
# SAMPLE TASK
# ============================================================

task = EnterpriseTask(
    employee_name="Ahmed Hassan",
    department="Finance",
    request_type="Cloud Infrastructure Purchase",
    amount=15000
)

#%%
# ============================================================
# RUN WORKFLOW
# ============================================================

workflow = SequentialEnterpriseWorkflow()

workflow.run(task)

#%%
# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("================================================")
print("FINAL ENTERPRISE REPORT")
print("================================================")

print(f"Employee Name : {task.employee_name}")
print(f"Department    : {task.department}")
print(f"Request Type  : {task.request_type}")
print(f"Amount        : {task.amount}")

print(f"Approved      : {task.approved}")
print(f"Rejected      : {task.rejected}")

print(f"Execution     : {task.execution_status}")

print("================================================")

#%%
# ============================================================
# DISPLAY AUDIT LOGS
# ============================================================

print("\n")
print("================================================")
print("AUDIT LOGS")
print("================================================")

for log in task.logs:
    print(log)