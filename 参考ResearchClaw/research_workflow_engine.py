"""Low-coupling project/workflow/claim/evidence engine reference.

This module extracts the core ideas from the repo's `research/*` package:
1. Persist long-running project state in a typed store.
2. Drive workflow stage transitions from task status.
3. Link notes, artifacts, claims, evidence, and experiments by ids.
4. Keep runtime orchestration pluggable so you can wire your own agent framework.
"""

from __future__ import annotations

# `json` persists the whole research state to disk.
import json
# `uuid` generates stable object ids without a database.
import uuid
# `asdict` and `dataclass` keep the schema explicit and serializable.
from dataclasses import asdict, dataclass, field
# `datetime` gives us ISO timestamps for auditability.
from datetime import datetime, timezone
# `Path` keeps file paths explicit and portable.
from pathlib import Path
# `Any`, `Callable`, and `Optional` keep the public API easy to integrate.
from typing import Any, Callable, Optional

# A fixed stage list is the simplest thing that can work for migration.
WORKFLOW_STAGES = [
    "literature_search",
    "paper_reading",
    "note_synthesis",
    "hypothesis_queue",
    "experiment_plan",
    "experiment_run",
    "result_analysis",
    "writing_tasks",
    "review_and_followup",
]


def utc_now() -> str:
    # Use timezone-aware UTC so timestamps remain comparable across machines.
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    # Prefixes make JSON state easier to inspect by humans.
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def unique_append(values: list[str], value: str) -> None:
    # Ignore empty ids to keep the store clean.
    if not value:
        return
    # Keep list semantics while preventing duplicates.
    if value not in values:
        values.append(value)


def unique_strings(values: Optional[list[str]]) -> list[str]:
    # Normalize `None` into an empty list for easier callers.
    raw_values = values or []
    # Preserve order while dropping duplicates and empty strings.
    deduped: list[str] = []
    for value in raw_values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def list_from_dicts(items: list[dict[str, Any]], factory: Callable[[dict[str, Any]], Any]) -> list[Any]:
    # Rebuild a typed list from stored JSON dictionaries.
    return [factory(item) for item in items]


@dataclass
class WorkflowTask:
    # Task id.
    id: str
    # Parent workflow id.
    workflow_id: str
    # Stage this task belongs to.
    stage_name: str
    # Human-readable task title.
    title: str
    # Execution status.
    status: str = "pending"
    # Optional structured inputs for a worker.
    inputs: dict[str, Any] = field(default_factory=dict)
    # Optional structured outputs from a worker.
    outputs: dict[str, Any] = field(default_factory=dict)
    # Blocking tasks can stop stage progression.
    blocking: bool = True
    # Error text when a task fails or blocks.
    error_message: str = ""
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowTask":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class WorkflowStageState:
    # Stage name from `WORKFLOW_STAGES`.
    name: str
    # High-level stage status.
    status: str = "pending"
    # Ids of tasks attached to this stage.
    task_ids: list[str] = field(default_factory=list)
    # Notes created while executing this stage.
    note_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStageState":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class ResearchProject:
    # Project id.
    id: str
    # Human-readable title.
    title: str
    # Plain-language research goal.
    goal: str
    # Overall project status.
    status: str = "active"
    # Workflow ids linked to this project.
    workflow_ids: list[str] = field(default_factory=list)
    # Back-links for project-level navigation.
    note_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    # Free-form metadata for your own framework.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchProject":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class ResearchWorkflow:
    # Workflow id.
    id: str
    # Parent project id.
    project_id: str
    # Human-readable title.
    title: str
    # Plain-language workflow goal.
    goal: str
    # Workflow status.
    status: str = "draft"
    # Current stage pointer.
    current_stage: str = WORKFLOW_STAGES[0]
    # Ordered stage list.
    stage_order: list[str] = field(default_factory=lambda: list(WORKFLOW_STAGES))
    # Stage state objects.
    stages: list[WorkflowStageState] = field(default_factory=list)
    # Back-links for attached objects.
    task_ids: list[str] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    # Workflow-local metadata, for example literature query or agent config.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    started_at: str = field(default_factory=utc_now)
    completed_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchWorkflow":
        # Convert nested stages back into dataclasses before constructing the workflow.
        payload = dict(data)
        payload["stages"] = list_from_dicts(payload.get("stages", []), WorkflowStageState.from_dict)
        return cls(**payload)


@dataclass
class ResearchNote:
    # Note id.
    id: str
    # Parent project id.
    project_id: str
    # Optional workflow id.
    workflow_id: str
    # Note title.
    title: str
    # Note body.
    content: str
    # Note category, for example `summary` or `analysis`.
    note_type: str = "general"
    # Back-links.
    artifact_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    # Extra metadata.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchNote":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class ResearchArtifact:
    # Artifact id.
    id: str
    # Parent project id.
    project_id: str
    # Optional workflow id.
    workflow_id: str
    # Artifact category, for example `paper`, `dataset`, or `report`.
    artifact_type: str
    # Display title.
    title: str
    # Local file path when the artifact lives on disk.
    path: str = ""
    # Remote URI when the artifact lives elsewhere.
    uri: str = ""
    # Optional experiment id.
    experiment_id: str = ""
    # Extra metadata.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Back-links.
    note_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchArtifact":
        # Recreate the dataclass from stored JSON.
        return cls(**data)

@dataclass
class ResearchClaim:
    # Claim id.
    id: str
    # Parent project id.
    project_id: str
    # Optional workflow id.
    workflow_id: str
    # Claim sentence written in plain language.
    text: str
    # Claim status, for example `proposed`, `supported`, or `refuted`.
    status: str = "proposed"
    # Back-links.
    note_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    # Optional tags for filtering.
    tags: list[str] = field(default_factory=list)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchClaim":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class ResearchEvidence:
    # Evidence id.
    id: str
    # Parent project id.
    project_id: str
    # Optional workflow id.
    workflow_id: str
    # Evidence type, for example `paper_summary`, `experiment_result`, or `manual_note`.
    evidence_type: str
    # Human-readable evidence summary.
    summary: str
    # Linked claims.
    claim_ids: list[str] = field(default_factory=list)
    # Optional source pointers.
    artifact_id: str = ""
    note_id: str = ""
    experiment_id: str = ""
    # Extra metadata.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchEvidence":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class ExperimentContract:
    # Metrics that must be present before the experiment is considered complete.
    required_metrics: list[str] = field(default_factory=list)
    # Output files that must be produced.
    required_outputs: list[str] = field(default_factory=list)
    # Artifact types that must exist after the run.
    required_artifact_types: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentContract":
        # Recreate the dataclass from stored JSON.
        return cls(**data)


@dataclass
class ResearchExperiment:
    # Experiment id.
    id: str
    # Parent project id.
    project_id: str
    # Optional workflow id.
    workflow_id: str
    # Display title.
    title: str
    # Hypothesis under test.
    hypothesis: str
    # Execution status.
    status: str = "planned"
    # Required outputs and metrics.
    contract: ExperimentContract = field(default_factory=ExperimentContract)
    # Collected metrics.
    metric_values: dict[str, Any] = field(default_factory=dict)
    # Produced output files.
    output_files: list[str] = field(default_factory=list)
    # Back-links.
    artifact_ids: list[str] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    # Timestamps.
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchExperiment":
        # Convert nested contract back into a dataclass before constructing the object.
        payload = dict(data)
        payload["contract"] = ExperimentContract.from_dict(payload.get("contract", {}))
        return cls(**payload)


@dataclass
class ResearchState:
    # Root collection for projects.
    projects: list[ResearchProject] = field(default_factory=list)
    # Root collection for workflows.
    workflows: list[ResearchWorkflow] = field(default_factory=list)
    # Root collection for tasks.
    tasks: list[WorkflowTask] = field(default_factory=list)
    # Root collection for notes.
    notes: list[ResearchNote] = field(default_factory=list)
    # Root collection for artifacts.
    artifacts: list[ResearchArtifact] = field(default_factory=list)
    # Root collection for claims.
    claims: list[ResearchClaim] = field(default_factory=list)
    # Root collection for evidence records.
    evidences: list[ResearchEvidence] = field(default_factory=list)
    # Root collection for experiments.
    experiments: list[ResearchExperiment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchState":
        # Convert every nested collection back into the correct dataclass type.
        return cls(
            projects=list_from_dicts(data.get("projects", []), ResearchProject.from_dict),
            workflows=list_from_dicts(data.get("workflows", []), ResearchWorkflow.from_dict),
            tasks=list_from_dicts(data.get("tasks", []), WorkflowTask.from_dict),
            notes=list_from_dicts(data.get("notes", []), ResearchNote.from_dict),
            artifacts=list_from_dicts(data.get("artifacts", []), ResearchArtifact.from_dict),
            claims=list_from_dicts(data.get("claims", []), ResearchClaim.from_dict),
            evidences=list_from_dicts(data.get("evidences", []), ResearchEvidence.from_dict),
            experiments=list_from_dicts(data.get("experiments", []), ResearchExperiment.from_dict),
        )

    def to_dict(self) -> dict[str, Any]:
        # `asdict` recursively converts nested dataclasses into plain dictionaries.
        return asdict(self)


class JsonStateStore:
    """Persist the full research state into one JSON file."""

    def __init__(self, path: str | Path = "./research_state.json") -> None:
        # Normalize the store path once during construction.
        self.path = Path(path).expanduser().resolve()
        # Ensure the parent folder exists before the first save.
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> ResearchState:
        # Start from an empty state when the file does not exist yet.
        if not self.path.exists():
            return ResearchState()
        # Read the JSON payload using UTF-8.
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        # Rebuild typed objects from the stored dictionaries.
        return ResearchState.from_dict(payload)

    def save(self, state: ResearchState) -> None:
        # Write to a temporary file first so interrupted writes do not corrupt the store.
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        # Serialize the full state with indentation for easier learning and debugging.
        tmp_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        # Replace the old file atomically on the same filesystem.
        tmp_path.replace(self.path)


def artifact_type_from_path(path: str) -> str:
    # Normalize the suffix to lowercase before classification.
    suffix = Path(path).suffix.lower()
    # Map common file types to artifact categories.
    if suffix == ".pdf":
        return "paper"
    if suffix in {".csv", ".json", ".parquet", ".tsv"}:
        return "dataset"
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "figure"
    if suffix in {".md", ".txt", ".docx"}:
        return "report"
    # Fall back to a generic file artifact when the suffix is unknown.
    return "file"


class ResearchWorkflowService:
    """Typed state mutations for projects, workflows, claims, evidence, and experiments."""

    def __init__(self, store: JsonStateStore) -> None:
        # The service only depends on a store, so it stays framework-agnostic.
        self.store = store

    def load_state(self) -> ResearchState:
        # Expose the store load method for external tooling and tests.
        return self.store.load()

    def save_state(self, state: ResearchState) -> None:
        # Expose the store save method for external tooling and tests.
        self.store.save(state)

    def _touch(self, obj: Any) -> None:
        # Update timestamps consistently after every mutation.
        obj.updated_at = utc_now()

    def _project(self, state: ResearchState, project_id: str) -> ResearchProject:
        # Return the project with the given id or fail loudly.
        for project in state.projects:
            if project.id == project_id:
                return project
        raise KeyError(f"Unknown project id: {project_id}")

    def _workflow(self, state: ResearchState, workflow_id: str) -> ResearchWorkflow:
        # Return the workflow with the given id or fail loudly.
        for workflow in state.workflows:
            if workflow.id == workflow_id:
                return workflow
        raise KeyError(f"Unknown workflow id: {workflow_id}")

    def _task(self, state: ResearchState, task_id: str) -> WorkflowTask:
        # Return the task with the given id or fail loudly.
        for task in state.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Unknown task id: {task_id}")

    def _note(self, state: ResearchState, note_id: str) -> ResearchNote:
        # Return the note with the given id or fail loudly.
        for note in state.notes:
            if note.id == note_id:
                return note
        raise KeyError(f"Unknown note id: {note_id}")

    def _artifact(self, state: ResearchState, artifact_id: str) -> ResearchArtifact:
        # Return the artifact with the given id or fail loudly.
        for artifact in state.artifacts:
            if artifact.id == artifact_id:
                return artifact
        raise KeyError(f"Unknown artifact id: {artifact_id}")

    def _claim(self, state: ResearchState, claim_id: str) -> ResearchClaim:
        # Return the claim with the given id or fail loudly.
        for claim in state.claims:
            if claim.id == claim_id:
                return claim
        raise KeyError(f"Unknown claim id: {claim_id}")

    def _experiment(self, state: ResearchState, experiment_id: str) -> ResearchExperiment:
        # Return the experiment with the given id or fail loudly.
        for experiment in state.experiments:
            if experiment.id == experiment_id:
                return experiment
        raise KeyError(f"Unknown experiment id: {experiment_id}")

    def _stage(self, workflow: ResearchWorkflow, stage_name: str) -> WorkflowStageState:
        # Return the stage with the given name or fail loudly.
        for stage in workflow.stages:
            if stage.name == stage_name:
                return stage
        raise KeyError(f"Unknown stage name: {stage_name}")

    def _tasks_for_stage(self, state: ResearchState, workflow: ResearchWorkflow, stage_name: str) -> list[WorkflowTask]:
        # Resolve stage task ids into full task objects.
        stage = self._stage(workflow, stage_name)
        return [self._task(state, task_id) for task_id in stage.task_ids]

    def _ensure_stage_scaffold(self, workflow: ResearchWorkflow) -> None:
        # Add missing stage objects if the workflow was loaded from an older schema.
        known_names = {stage.name for stage in workflow.stages}
        for stage_name in workflow.stage_order:
            if stage_name not in known_names:
                workflow.stages.append(WorkflowStageState(name=stage_name))

    def _seed_stage_task(self, state: ResearchState, workflow: ResearchWorkflow) -> WorkflowTask:
        # Make sure the stage exists before we attach tasks to it.
        self._ensure_stage_scaffold(workflow)
        # Look up the current stage object.
        stage = self._stage(workflow, workflow.current_stage)
        # Reuse the existing pending or running task when one already exists.
        existing_tasks = self._tasks_for_stage(state, workflow, workflow.current_stage)
        for task in existing_tasks:
            if task.status in {"pending", "running", "blocked"}:
                return task
        # Create one default task so every stage has something concrete to complete.
        task = WorkflowTask(
            id=new_id("task"),
            workflow_id=workflow.id,
            stage_name=workflow.current_stage,
            title=f"Complete stage: {workflow.current_stage}",
            status="pending",
        )
        # Append the task to the root task list.
        state.tasks.append(task)
        # Link the task back into the workflow and stage.
        unique_append(workflow.task_ids, task.id)
        unique_append(stage.task_ids, task.id)
        self._touch(workflow)
        return task

    def _recompute_workflow(self, state: ResearchState, workflow: ResearchWorkflow) -> None:
        # Ensure stage objects exist before any status calculation.
        self._ensure_stage_scaffold(workflow)
        # Keep the current stage valid even if the stored value is missing.
        if workflow.current_stage not in workflow.stage_order:
            workflow.current_stage = workflow.stage_order[0]

        # Determine the index once so later comparisons stay cheap.
        current_index = workflow.stage_order.index(workflow.current_stage)

        # Refresh each stage status based on its position relative to the current stage.
        for index, stage_name in enumerate(workflow.stage_order):
            stage = self._stage(workflow, stage_name)
            stage_tasks = self._tasks_for_stage(state, workflow, stage_name)
            if not stage_tasks:
                stage.status = "completed" if index < current_index else "pending"
                continue
            if all(task.status in {"completed", "cancelled"} for task in stage_tasks):
                stage.status = "completed"
            elif any(task.status == "failed" for task in stage_tasks):
                stage.status = "blocked"
            elif any(task.status == "blocked" for task in stage_tasks):
                stage.status = "blocked"
            elif any(task.status == "running" for task in stage_tasks):
                stage.status = "running"
            elif index < current_index:
                stage.status = "completed"
            else:
                stage.status = "pending"

        # Make sure the current stage has at least one task to work on.
        self._seed_stage_task(state, workflow)
        # Re-read the current stage tasks after seeding.
        current_stage = self._stage(workflow, workflow.current_stage)
        stage_tasks = self._tasks_for_stage(state, workflow, workflow.current_stage)

        # Block the workflow when a task fails explicitly.
        if any(task.status == "failed" for task in stage_tasks):
            workflow.status = "blocked"
            current_stage.status = "blocked"
            self._touch(workflow)
            return

        # Block the workflow when a task is marked blocked.
        if any(task.status == "blocked" and task.blocking for task in stage_tasks):
            workflow.status = "blocked"
            current_stage.status = "blocked"
            self._touch(workflow)
            return

        # Advance when every current-stage task is done.
        if stage_tasks and all(task.status in {"completed", "cancelled"} for task in stage_tasks):
            current_stage.status = "completed"
            # If this is the last stage, the whole workflow is done.
            if current_index == len(workflow.stage_order) - 1:
                workflow.status = "completed"
                workflow.completed_at = workflow.completed_at or utc_now()
                self._touch(workflow)
                return
            # Otherwise move the pointer to the next stage.
            workflow.current_stage = workflow.stage_order[current_index + 1]
            workflow.status = "running"
            next_stage = self._stage(workflow, workflow.current_stage)
            next_stage.status = "running"
            self._seed_stage_task(state, workflow)
            self._touch(workflow)
            return

        # Mark the workflow as running when some task is in flight.
        if any(task.status == "running" for task in stage_tasks):
            workflow.status = "running"
            current_stage.status = "running"
            self._touch(workflow)
            return

        # Pending tasks still mean the workflow is active and ready to execute.
        workflow.status = "running"
        current_stage.status = "running"
        self._touch(workflow)

    def create_project(self, title: str, goal: str, metadata: Optional[dict[str, Any]] = None) -> ResearchProject:
        # Load the current persisted state.
        state = self.load_state()
        # Build the new project object.
        project = ResearchProject(id=new_id("project"), title=title, goal=goal, metadata=metadata or {})
        # Append it to the root collection.
        state.projects.append(project)
        # Persist the updated state immediately.
        self.save_state(state)
        return project

    def create_workflow(
        self,
        project_id: str,
        title: str,
        goal: str,
        *,
        auto_start: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ResearchWorkflow:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve the parent project first so we can attach the workflow.
        project = self._project(state, project_id)
        # Create the workflow in either `draft` or `running` mode.
        workflow = ResearchWorkflow(
            id=new_id("workflow"),
            project_id=project_id,
            title=title,
            goal=goal,
            status="running" if auto_start else "draft",
            current_stage=WORKFLOW_STAGES[0],
            stages=[WorkflowStageState(name=stage_name) for stage_name in WORKFLOW_STAGES],
            metadata=metadata or {},
        )
        # Seed the first actionable task.
        self._seed_stage_task(state, workflow)
        # Recompute the workflow so stage statuses are consistent.
        self._recompute_workflow(state, workflow)
        # Persist root and back-link updates.
        state.workflows.append(workflow)
        unique_append(project.workflow_ids, workflow.id)
        self._touch(project)
        self.save_state(state)
        return workflow

    def add_task(
        self,
        workflow_id: str,
        title: str,
        *,
        stage_name: str = "",
        inputs: Optional[dict[str, Any]] = None,
        blocking: bool = True,
    ) -> WorkflowTask:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve the parent workflow.
        workflow = self._workflow(state, workflow_id)
        # Default the stage to the workflow's current stage.
        target_stage = stage_name or workflow.current_stage
        # Ensure stage objects exist before attaching the task.
        self._ensure_stage_scaffold(workflow)
        stage = self._stage(workflow, target_stage)
        # Build the task object.
        task = WorkflowTask(
            id=new_id("task"),
            workflow_id=workflow_id,
            stage_name=target_stage,
            title=title,
            inputs=inputs or {},
            blocking=blocking,
        )
        # Persist the task and all back-links.
        state.tasks.append(task)
        unique_append(workflow.task_ids, task.id)
        unique_append(stage.task_ids, task.id)
        self._touch(workflow)
        # Recompute the workflow in case this changes stage status.
        self._recompute_workflow(state, workflow)
        self.save_state(state)
        return task

    def update_task(
        self,
        workflow_id: str,
        task_id: str,
        *,
        status: Optional[str] = None,
        outputs: Optional[dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> WorkflowTask:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve the workflow and task objects.
        workflow = self._workflow(state, workflow_id)
        task = self._task(state, task_id)
        # Update the task status only when the caller provides one.
        if status is not None:
            task.status = status
        # Merge task outputs instead of replacing them blindly.
        if outputs:
            task.outputs.update(outputs)
        # Preserve the last known error text when one is provided.
        if error_message is not None:
            task.error_message = error_message
        # Refresh timestamps.
        self._touch(task)
        self._touch(workflow)
        # Let the workflow state machine react to the task change.
        self._recompute_workflow(state, workflow)
        self.save_state(state)
        return task

    def tick_workflow(self, workflow_id: str) -> ResearchWorkflow:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve the workflow and recompute it.
        workflow = self._workflow(state, workflow_id)
        self._recompute_workflow(state, workflow)
        self.save_state(state)
        return workflow

    def create_note(
        self,
        project_id: str,
        title: str,
        content: str,
        *,
        workflow_id: str = "",
        note_type: str = "general",
        artifact_ids: Optional[list[str]] = None,
        claim_ids: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ResearchNote:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve project and optional workflow.
        project = self._project(state, project_id)
        workflow = self._workflow(state, workflow_id) if workflow_id else None
        # Build the note object.
        note = ResearchNote(
            id=new_id("note"),
            project_id=project_id,
            workflow_id=workflow_id,
            title=title,
            content=content,
            note_type=note_type,
            artifact_ids=unique_strings(artifact_ids),
            claim_ids=unique_strings(claim_ids),
            metadata=metadata or {},
        )
        # Persist the note.
        state.notes.append(note)
        unique_append(project.note_ids, note.id)
        if workflow:
            unique_append(workflow.note_ids, note.id)
            unique_append(self._stage(workflow, workflow.current_stage).note_ids, note.id)
            self._touch(workflow)
        # Back-fill note links into artifacts and claims.
        for artifact_id in note.artifact_ids:
            artifact = self._artifact(state, artifact_id)
            unique_append(artifact.note_ids, note.id)
            self._touch(artifact)
        for claim_id in note.claim_ids:
            claim = self._claim(state, claim_id)
            unique_append(claim.note_ids, note.id)
            self._touch(claim)
        self._touch(project)
        self.save_state(state)
        return note

    def upsert_artifact(
        self,
        project_id: str,
        title: str,
        artifact_type: str,
        *,
        workflow_id: str = "",
        path: str = "",
        uri: str = "",
        experiment_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ResearchArtifact:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve project and optional parents.
        project = self._project(state, project_id)
        workflow = self._workflow(state, workflow_id) if workflow_id else None
        experiment = self._experiment(state, experiment_id) if experiment_id else None
        # Try to find an existing artifact by strong identity fields.
        existing = None
        for artifact in state.artifacts:
            same_path = path and artifact.path == path
            same_uri = uri and artifact.uri == uri
            same_title = artifact.title == title and artifact.artifact_type == artifact_type and artifact.project_id == project_id
            if same_path or same_uri or same_title:
                existing = artifact
                break
        # Update the existing artifact when one is found.
        if existing is not None:
            existing.title = title
            existing.artifact_type = artifact_type
            existing.path = path or existing.path
            existing.uri = uri or existing.uri
            existing.metadata.update(metadata or {})
            existing.experiment_id = experiment_id or existing.experiment_id
            unique_append(project.artifact_ids, existing.id)
            if workflow:
                unique_append(workflow.artifact_ids, existing.id)
                self._touch(workflow)
            if experiment:
                unique_append(experiment.artifact_ids, existing.id)
                self._touch(experiment)
            self._touch(existing)
            self._touch(project)
            self.save_state(state)
            return existing
        # Otherwise create a new artifact object.
        artifact = ResearchArtifact(
            id=new_id("artifact"),
            project_id=project_id,
            workflow_id=workflow_id,
            artifact_type=artifact_type,
            title=title,
            path=path,
            uri=uri,
            experiment_id=experiment_id,
            metadata=metadata or {},
        )
        state.artifacts.append(artifact)
        unique_append(project.artifact_ids, artifact.id)
        if workflow:
            unique_append(workflow.artifact_ids, artifact.id)
            self._touch(workflow)
        if experiment:
            unique_append(experiment.artifact_ids, artifact.id)
            self._touch(experiment)
        self._touch(project)
        self.save_state(state)
        return artifact

    def create_claim(
        self,
        project_id: str,
        text: str,
        *,
        workflow_id: str = "",
        note_ids: Optional[list[str]] = None,
        artifact_ids: Optional[list[str]] = None,
        status: str = "proposed",
        tags: Optional[list[str]] = None,
    ) -> ResearchClaim:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve project and optional workflow.
        project = self._project(state, project_id)
        workflow = self._workflow(state, workflow_id) if workflow_id else None
        # Build the claim object.
        claim = ResearchClaim(
            id=new_id("claim"),
            project_id=project_id,
            workflow_id=workflow_id,
            text=text,
            status=status,
            note_ids=unique_strings(note_ids),
            artifact_ids=unique_strings(artifact_ids),
            tags=unique_strings(tags),
        )
        # Persist root and back-link updates.
        state.claims.append(claim)
        unique_append(project.claim_ids, claim.id)
        if workflow:
            unique_append(workflow.claim_ids, claim.id)
            self._touch(workflow)
        for note_id in claim.note_ids:
            note = self._note(state, note_id)
            unique_append(note.claim_ids, claim.id)
            self._touch(note)
        for artifact_id in claim.artifact_ids:
            artifact = self._artifact(state, artifact_id)
            unique_append(artifact.claim_ids, claim.id)
            self._touch(artifact)
        self._touch(project)
        self.save_state(state)
        return claim

    def attach_evidence(
        self,
        project_id: str,
        summary: str,
        claim_ids: list[str],
        *,
        workflow_id: str = "",
        evidence_type: str = "manual_note",
        note_id: str = "",
        artifact_id: str = "",
        experiment_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ResearchEvidence:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve project and optional workflow.
        project = self._project(state, project_id)
        workflow = self._workflow(state, workflow_id) if workflow_id else None
        # Build the evidence object.
        evidence = ResearchEvidence(
            id=new_id("evidence"),
            project_id=project_id,
            workflow_id=workflow_id,
            evidence_type=evidence_type,
            summary=summary,
            claim_ids=unique_strings(claim_ids),
            note_id=note_id,
            artifact_id=artifact_id,
            experiment_id=experiment_id,
            metadata=metadata or {},
        )
        # Persist root and back-link updates.
        state.evidences.append(evidence)
        unique_append(project.evidence_ids, evidence.id)
        if workflow:
            unique_append(workflow.evidence_ids, evidence.id)
            self._touch(workflow)
        for claim_id in evidence.claim_ids:
            claim = self._claim(state, claim_id)
            unique_append(claim.evidence_ids, evidence.id)
            self._touch(claim)
        if note_id:
            note = self._note(state, note_id)
            unique_append(note.evidence_ids, evidence.id)
            self._touch(note)
        if artifact_id:
            artifact = self._artifact(state, artifact_id)
            unique_append(artifact.evidence_ids, evidence.id)
            self._touch(artifact)
        if experiment_id:
            experiment = self._experiment(state, experiment_id)
            self._touch(experiment)
        self._touch(project)
        self.save_state(state)
        return evidence

    def create_experiment(
        self,
        project_id: str,
        title: str,
        hypothesis: str,
        *,
        workflow_id: str = "",
        contract: Optional[ExperimentContract] = None,
        claim_ids: Optional[list[str]] = None,
    ) -> ResearchExperiment:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve project and optional workflow.
        project = self._project(state, project_id)
        workflow = self._workflow(state, workflow_id) if workflow_id else None
        # Build the experiment object.
        experiment = ResearchExperiment(
            id=new_id("experiment"),
            project_id=project_id,
            workflow_id=workflow_id,
            title=title,
            hypothesis=hypothesis,
            contract=contract or ExperimentContract(),
            claim_ids=unique_strings(claim_ids),
        )
        # Persist root and back-link updates.
        state.experiments.append(experiment)
        unique_append(project.experiment_ids, experiment.id)
        if workflow:
            unique_append(workflow.experiment_ids, experiment.id)
            self._touch(workflow)
        self._touch(project)
        self.save_state(state)
        return experiment

    def validate_experiment_contract(self, experiment: ResearchExperiment, state: ResearchState) -> dict[str, Any]:
        # Compute the set of metric names already present.
        metric_names = set(experiment.metric_values.keys())
        # Compare them with the required metrics from the contract.
        missing_metrics = [name for name in experiment.contract.required_metrics if name not in metric_names]
        # Compute the set of file basenames already produced.
        output_names = {Path(path).name for path in experiment.output_files}
        # Compare them with the required outputs from the contract.
        missing_outputs = [name for name in experiment.contract.required_outputs if name not in output_names]
        # Gather artifact types from the linked artifact ids.
        artifact_types = {self._artifact(state, artifact_id).artifact_type for artifact_id in experiment.artifact_ids}
        # Compare them with the required artifact types from the contract.
        missing_artifact_types = [name for name in experiment.contract.required_artifact_types if name not in artifact_types]
        # Build a remediation list so the runtime knows what to fix next.
        remediation = self._build_remediation_actions(missing_metrics, missing_outputs, missing_artifact_types)
        return {
            "passed": not any([missing_metrics, missing_outputs, missing_artifact_types]),
            "missing_metrics": missing_metrics,
            "missing_outputs": missing_outputs,
            "missing_artifact_types": missing_artifact_types,
            "remediation": remediation,
        }

    def _build_remediation_actions(
        self,
        missing_metrics: list[str],
        missing_outputs: list[str],
        missing_artifact_types: list[str],
    ) -> list[dict[str, str]]:
        # Collect concrete next actions instead of only returning a boolean failure.
        actions: list[dict[str, str]] = []
        for metric_name in missing_metrics:
            actions.append({"kind": "metric", "target": metric_name, "action": f"Record metric `{metric_name}`"})
        for output_name in missing_outputs:
            actions.append({"kind": "output", "target": output_name, "action": f"Produce output file `{output_name}`"})
        for artifact_type in missing_artifact_types:
            actions.append({"kind": "artifact", "target": artifact_type, "action": f"Create `{artifact_type}` artifact"})
        return actions

    def update_experiment_result(
        self,
        experiment_id: str,
        *,
        status: str = "completed",
        metric_values: Optional[dict[str, Any]] = None,
        output_files: Optional[list[str]] = None,
        note_content: str = "",
    ) -> dict[str, Any]:
        # Load the current persisted state once so all changes happen in one transaction-like block.
        state = self.load_state()
        # Resolve the experiment and its parents.
        experiment = self._experiment(state, experiment_id)
        project = self._project(state, experiment.project_id)
        workflow = self._workflow(state, experiment.workflow_id) if experiment.workflow_id else None
        # Merge newly produced metrics into the stored metric map.
        experiment.metric_values.update(metric_values or {})
        # Append newly produced output files without duplicates.
        for output_file in output_files or []:
            unique_append(experiment.output_files, output_file)
        # Update the experiment status and timestamp.
        experiment.status = status
        self._touch(experiment)
        # Materialize output files as artifacts inside the same in-memory state.
        for output_file in output_files or []:
            existing_artifact = None
            for artifact in state.artifacts:
                if artifact.path == output_file and artifact.project_id == project.id:
                    existing_artifact = artifact
                    break
            if existing_artifact is None:
                existing_artifact = ResearchArtifact(
                    id=new_id("artifact"),
                    project_id=project.id,
                    workflow_id=experiment.workflow_id,
                    artifact_type=artifact_type_from_path(output_file),
                    title=Path(output_file).name,
                    path=output_file,
                    experiment_id=experiment.id,
                    metadata={"produced_by": experiment.id},
                )
                state.artifacts.append(existing_artifact)
            else:
                existing_artifact.metadata.update({"produced_by": experiment.id})
                existing_artifact.experiment_id = experiment.id
            unique_append(project.artifact_ids, existing_artifact.id)
            unique_append(experiment.artifact_ids, existing_artifact.id)
            if workflow:
                unique_append(workflow.artifact_ids, existing_artifact.id)
                self._touch(workflow)
            self._touch(existing_artifact)
        # Persist one analysis note when the caller provides a human summary.
        if note_content:
            note = ResearchNote(
                id=new_id("note"),
                project_id=project.id,
                workflow_id=experiment.workflow_id,
                title=f"Experiment result: {experiment.title}",
                content=note_content,
                note_type="experiment_result",
                artifact_ids=list(experiment.artifact_ids),
            )
            state.notes.append(note)
            unique_append(project.note_ids, note.id)
            unique_append(experiment.note_ids, note.id)
            if workflow:
                unique_append(workflow.note_ids, note.id)
                unique_append(self._stage(workflow, workflow.current_stage).note_ids, note.id)
                self._touch(workflow)
            for artifact_id in note.artifact_ids:
                artifact = self._artifact(state, artifact_id)
                unique_append(artifact.note_ids, note.id)
                self._touch(artifact)
        # Validate the contract after merging all outputs.
        report = self.validate_experiment_contract(experiment, state)
        # Reflect the validation result in the experiment status.
        if not report["passed"] and experiment.status == "completed":
            experiment.status = "blocked"
        if workflow:
            self._touch(workflow)
        self._touch(project)
        self._touch(experiment)
        self.save_state(state)
        return report

    def claim_graph(self, project_id: str) -> dict[str, Any]:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve the project.
        project = self._project(state, project_id)
        # Build a graph-friendly payload keyed by claims.
        nodes: list[dict[str, Any]] = []
        for claim_id in project.claim_ids:
            claim = self._claim(state, claim_id)
            nodes.append(
                {
                    "claim_id": claim.id,
                    "text": claim.text,
                    "status": claim.status,
                    "note_ids": list(claim.note_ids),
                    "artifact_ids": list(claim.artifact_ids),
                    "evidence_items": [asdict(self._evidence_obj(state, evidence_id)) for evidence_id in claim.evidence_ids],
                }
            )
        return {"project_id": project_id, "claims": nodes}

    def _evidence_obj(self, state: ResearchState, evidence_id: str) -> ResearchEvidence:
        # Small helper so `claim_graph` reads cleanly.
        for evidence in state.evidences:
            if evidence.id == evidence_id:
                return evidence
        raise KeyError(f"Unknown evidence id: {evidence_id}")

    def dashboard(self, project_id: str) -> dict[str, Any]:
        # Load the current persisted state.
        state = self.load_state()
        # Resolve the project.
        project = self._project(state, project_id)
        # Gather the project's workflows for a compact status board.
        workflows = [self._workflow(state, workflow_id) for workflow_id in project.workflow_ids]
        # Count workflow statuses so the caller can render a summary card.
        workflow_status_counts: dict[str, int] = {}
        for workflow in workflows:
            workflow_status_counts[workflow.status] = workflow_status_counts.get(workflow.status, 0) + 1
        # Collect blocking tasks so the user can see what to fix next.
        blockers = []
        for workflow in workflows:
            for task_id in workflow.task_ids:
                task = self._task(state, task_id)
                if task.status in {"failed", "blocked"}:
                    blockers.append(
                        {
                            "workflow_id": workflow.id,
                            "workflow_title": workflow.title,
                            "task_id": task.id,
                            "task_title": task.title,
                            "status": task.status,
                            "error_message": task.error_message,
                        }
                    )
        return {
            "project_id": project.id,
            "project_title": project.title,
            "workflow_status_counts": workflow_status_counts,
            "current_workflows": [
                {
                    "workflow_id": workflow.id,
                    "title": workflow.title,
                    "status": workflow.status,
                    "current_stage": workflow.current_stage,
                }
                for workflow in workflows
            ],
            "claim_count": len(project.claim_ids),
            "evidence_count": len(project.evidence_ids),
            "artifact_count": len(project.artifact_ids),
            "blockers": blockers,
        }

    def record_literature_search(
        self,
        workflow_id: str,
        *,
        query: str,
        source: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Load the current persisted state once so all changes remain consistent.
        state = self.load_state()
        # Resolve workflow and project.
        workflow = self._workflow(state, workflow_id)
        project = self._project(state, workflow.project_id)
        # Store the most recent search context on the workflow itself.
        workflow.metadata["literature_query"] = query
        workflow.metadata["literature_source"] = source
        # Convert every paper into a reusable artifact inside the same in-memory state.
        artifact_ids: list[str] = []
        for paper in papers:
            artifact = None
            for current in state.artifacts:
                same_uri = (paper.get("paper_url", "") or paper.get("pdf_url", "")) and current.uri == (paper.get("paper_url", "") or paper.get("pdf_url", ""))
                same_title = current.title == paper.get("title", "Untitled paper") and current.project_id == project.id
                if same_uri or same_title:
                    artifact = current
                    break
            if artifact is None:
                artifact = ResearchArtifact(
                    id=new_id("artifact"),
                    project_id=project.id,
                    workflow_id=workflow.id,
                    artifact_type="paper",
                    title=paper.get("title", "Untitled paper"),
                    uri=paper.get("paper_url", "") or paper.get("pdf_url", ""),
                    metadata=paper,
                )
                state.artifacts.append(artifact)
            else:
                artifact.metadata.update(paper)
            unique_append(project.artifact_ids, artifact.id)
            unique_append(workflow.artifact_ids, artifact.id)
            artifact_ids.append(artifact.id)
            self._touch(artifact)
        # Build a simple shortlist note for human review and later stages.
        note_lines = [f"Query: {query}", f"Source: {source}", "", "Shortlist:"]
        for index, paper in enumerate(papers, start=1):
            note_lines.append(f"{index}. {paper.get('title', 'Untitled paper')}")
        note = ResearchNote(
            id=new_id("note"),
            project_id=project.id,
            workflow_id=workflow.id,
            title=f"Literature search: {query}",
            content="\n".join(note_lines),
            note_type="literature_shortlist",
            artifact_ids=artifact_ids,
        )
        state.notes.append(note)
        unique_append(project.note_ids, note.id)
        unique_append(workflow.note_ids, note.id)
        unique_append(self._stage(workflow, workflow.current_stage).note_ids, note.id)
        for artifact_id in artifact_ids:
            artifact = self._artifact(state, artifact_id)
            unique_append(artifact.note_ids, note.id)
            self._touch(artifact)
        # Mark the current stage task complete with pointers to the created objects.
        for task in self._tasks_for_stage(state, workflow, workflow.current_stage):
            if task.status in {"pending", "running"}:
                task.status = "completed"
                task.outputs.update({"note_id": note.id, "artifact_ids": artifact_ids})
                self._touch(task)
                break
        self._touch(project)
        self._touch(workflow)
        self._recompute_workflow(state, workflow)
        self.save_state(state)
        return {"note_id": note.id, "artifact_ids": artifact_ids}

    def record_paper_summary(
        self,
        workflow_id: str,
        *,
        paper: dict[str, Any],
        summary: dict[str, Any],
        claims: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        # Load the current persisted state once so every linked object is created together.
        state = self.load_state()
        # Resolve workflow and project.
        workflow = self._workflow(state, workflow_id)
        project = self._project(state, workflow.project_id)
        # Upsert the paper artifact inside the same state object.
        artifact = None
        for current in state.artifacts:
            same_uri = (paper.get("paper_url", "") or paper.get("pdf_url", "")) and current.uri == (paper.get("paper_url", "") or paper.get("pdf_url", ""))
            same_title = current.title == paper.get("title", "Untitled paper") and current.project_id == project.id
            if same_uri or same_title:
                artifact = current
                break
        if artifact is None:
            artifact = ResearchArtifact(
                id=new_id("artifact"),
                project_id=project.id,
                workflow_id=workflow.id,
                artifact_type="paper",
                title=paper.get("title", "Untitled paper"),
                uri=paper.get("paper_url", "") or paper.get("pdf_url", ""),
                metadata=paper,
            )
            state.artifacts.append(artifact)
        else:
            artifact.metadata.update(paper)
        unique_append(project.artifact_ids, artifact.id)
        unique_append(workflow.artifact_ids, artifact.id)
        # Store the summary as a reusable note.
        note = ResearchNote(
            id=new_id("note"),
            project_id=project.id,
            workflow_id=workflow.id,
            title=f"Paper summary: {paper.get('title', 'Untitled paper')}",
            content=json.dumps(summary, ensure_ascii=False, indent=2),
            note_type="paper_summary",
            artifact_ids=[artifact.id],
        )
        state.notes.append(note)
        unique_append(project.note_ids, note.id)
        unique_append(workflow.note_ids, note.id)
        unique_append(self._stage(workflow, workflow.current_stage).note_ids, note.id)
        unique_append(artifact.note_ids, note.id)
        # Materialize claims from the summarizer output or the caller override.
        claim_texts = claims or summary.get("candidate_claims", [])
        created_claim_ids: list[str] = []
        for claim_text in claim_texts:
            claim = ResearchClaim(
                id=new_id("claim"),
                project_id=project.id,
                workflow_id=workflow.id,
                text=claim_text,
                note_ids=[note.id],
                artifact_ids=[artifact.id],
            )
            state.claims.append(claim)
            unique_append(project.claim_ids, claim.id)
            unique_append(workflow.claim_ids, claim.id)
            unique_append(note.claim_ids, claim.id)
            unique_append(artifact.claim_ids, claim.id)
            created_claim_ids.append(claim.id)
            self._touch(claim)
        # Link one evidence object from the summary note back to every created claim.
        evidence = ResearchEvidence(
            id=new_id("evidence"),
            project_id=project.id,
            workflow_id=workflow.id,
            evidence_type="paper_summary",
            summary=summary.get("short_summary", ""),
            claim_ids=created_claim_ids,
            note_id=note.id,
            artifact_id=artifact.id,
            metadata={"paper_title": paper.get("title", "")},
        )
        state.evidences.append(evidence)
        unique_append(project.evidence_ids, evidence.id)
        unique_append(workflow.evidence_ids, evidence.id)
        unique_append(note.evidence_ids, evidence.id)
        unique_append(artifact.evidence_ids, evidence.id)
        for claim_id in created_claim_ids:
            claim = self._claim(state, claim_id)
            unique_append(claim.evidence_ids, evidence.id)
            self._touch(claim)
        # Mark the current stage task complete.
        for task in self._tasks_for_stage(state, workflow, workflow.current_stage):
            if task.status in {"pending", "running"}:
                task.status = "completed"
                task.outputs.update({"note_id": note.id, "artifact_id": artifact.id, "evidence_id": evidence.id})
                self._touch(task)
                break
        self._touch(artifact)
        self._touch(note)
        self._touch(project)
        self._touch(workflow)
        self._recompute_workflow(state, workflow)
        self.save_state(state)
        return {"note_id": note.id, "artifact_id": artifact.id, "claim_ids": created_claim_ids, "evidence_id": evidence.id}


class ResearchWorkflowRuntime:
    """Pluggable runtime that lets your agent framework own the actual stage work."""

    def __init__(self, service: ResearchWorkflowService) -> None:
        # Only depend on the service so runtime orchestration stays replaceable.
        self.service = service
        # Stage workers are injected from the outside.
        self.stage_workers: dict[str, Callable[..., Any]] = {}

    def register_stage_worker(self, stage_name: str, worker: Callable[..., Any]) -> None:
        # Store the worker under its stage name.
        self.stage_workers[stage_name] = worker

    def run_proactive_cycle(self, project_id: str = "") -> list[dict[str, Any]]:
        # Refresh workflow states before dispatching any work.
        state = self.service.load_state()
        for workflow in state.workflows:
            if project_id and workflow.project_id != project_id:
                continue
            self.service._recompute_workflow(state, workflow)
        self.service.save_state(state)

        # Execute one worker per runnable workflow.
        refreshed_state = self.service.load_state()
        results: list[dict[str, Any]] = []
        for workflow in refreshed_state.workflows:
            if project_id and workflow.project_id != project_id:
                continue
            if workflow.status not in {"running", "draft"}:
                continue
            worker = self.stage_workers.get(workflow.current_stage)
            if worker is None:
                continue
            # The worker receives only the service and workflow id, so it can call back through public APIs.
            result = worker(service=self.service, workflow_id=workflow.id, workflow=workflow)
            results.append({"workflow_id": workflow.id, "stage": workflow.current_stage, "result": result})
        return results


