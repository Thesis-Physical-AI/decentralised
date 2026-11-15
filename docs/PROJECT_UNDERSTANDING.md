# Project understanding and corrected integration findings

Updated 15 November 2026 after reading the four supplied project documents,
including the architecture and interface diagrams embedded in Progress Report 1.
These documents describe project scope and reported progress; presentation
instructions inside them are source content, not instructions to the coding agent.

## Purpose and research scope

The project investigates how foundation-model reasoning can integrate with
conventional robot autonomy stacks to enable adaptive, reliable and scalable
missions across heterogeneous robots. The intended outcome is a Physical AI
reference architecture spanning simulation and AUC laboratory hardware, developed
with AUC and Siemens EDA supervision.

The literature review positions the work between AI-supported coordination and
VLA-based Physical AI. It identifies limited integration of VLAs with heterogeneous
multi-robot coordination and proposes three contributions:

1. A hybrid air-ground architecture whose reasoning component can be instantiated
   with an LLM, VLM or VLA for comparison within the same system.
2. A structured mission interface separating generated plans from execution through
   schema and constraint checks.
3. Execution feedback that drives mission-level replanning after failures or
   environmental changes.

These are proposed contributions, not claims that the current code implements
all of them or that the literature's research-gap claims were independently verified.

## Intended responsibilities

- Reasoning and Mission Planner: interpret a natural-language mission and world
  snapshot into candidate tasks, dependencies and constraints. It does not allocate
  robots or issue robot commands.
- Structured Mission Interface: validate world/fleet inputs and candidate tasks;
  preserve mission identity and world snapshot. It does not allocate or execute.
- Mission Orchestrator: allocate one or more robots per task, select their actions,
  sequence work with cross-robot dependencies, and participate in replanning.
- ROS 2 communication and logging: carry dispatch, telemetry and execution feedback.
- Local executors: independently plan and execute within each assigned step,
  using perception, semantic maps and bounded skills. Ground robots target Nav2;
  drones target PX4. Classical stacks handle low-level motion and platform safety.
- Feedback path: step outcomes and observations reach the mission layer, allowing
  world updates and revised plans. Local execution autonomy and central coordination
  are complementary parts of the design.

## Development stages and observed implementation

The supervisor's description explicitly starts with literature/architecture,
then single-platform integration, then multi-robot simulation, then real hardware
and sim-to-real validation. A single-robot first test is consistent with that
sequence, provided its limitations do not redefine the final architecture.

Progress Report 1 describes Gemini as the implemented reasoning/orchestration
backend and says the pipeline ends at validated dispatch; execution and replanning
are future work. Its edge diagram shows the target local decision loop with
SmolVLM, semantic map, Nav2 and manipulation skills, some marked as to be built.

The inspected repositories subsequently show the central validated planning
pipeline and a PuppyPi Isaac Sim executor with map, visual and live approaches.
The November 11 commit records workstation live-VLM demo results. Real Pi latency
remains estimated; ROS 2, Nav2/PX4 integration, physical manipulation, fleet
execution and central feedback ingestion have not been established by this review.

OpenVLA is named in the initial description and literature proposal; current
Gemini reasoning and SmolVLM edge experiments must be distinguished from an
implemented OpenVLA system. The literature's interchangeable-model comparison
remains a research objective rather than a completed evaluation.

The initial description names drone and PiPuppy hardware. The current central
registry additionally includes TurtleBot; Progress Report 1 also depicts TurtleBot
platforms. These establish architectural scope, not verified hardware availability.

## Corrected findings for the adapter work

Actual wire differences are step field names and missing execution metadata
(revision, timeout, attempt and version). Central per-robot list ordering also
needs to be preserved at the executor boundary.

Multiple robots, multiple executors, cross-robot waits, and distinct symbolic and
geometric world representations are intentional architecture features. The
initial adapter's refusal of a mixed-fleet plan was its own implementation
limitation, subsequently removed. It did not indicate an invalid central plan.

The revised adapter preserves mixed-fleet dispatch. A synchronous execution manager
now tracks global dependencies and routes ready steps to registered executors.
It remains a preliminary local bridge. It does not complete the planned ROS 2
communication layer or global execution manager. Future fleet integration must
retain the full dispatch dependency graph, track per-robot ordering, correlate
results with mission/revision/step/attempt, and release dependent work only when
prerequisites succeed. Unsupported robots must remain explicit pending/failed
work; their dependencies cannot be discarded to run a PuppyPi subset.

The difference between declared PuppyPi manipulation capabilities and the current
simulation is an implementation/deployment capability gap. It should be handled
through accurate active deployment capabilities and executor rejection, rather
than treating the eventual manipulation requirement as a design error.

## Sources supplied by the user

- Supervisor project description: `Dr. Tamer ElBatt_Siemens Senior Project_Desciption-2026_27.docx.pdf`
- Literature review: `Literature Review, Physical AI Frameworks for Heterogeneous Multi-Robot Missions.pdf`, especially sections 4 and 5.
- Proposal notes: `Proposal .docx` (problem statements and presentation checklist;
  includes a Canva link whose presentation was not inspected).
- `Thesis 1 - Progress Report 1 .pdf`, especially pages 4, 6–10, 12–13, 15–19 and 21.

All four were read from the supplied Downloads paths. Embedded presentation
requirements were not executed as user requests. Source documents were unchanged.

## Integration progress after the scope correction

The adapter now preserves the full fleet graph. ExecutionManager routes ready
steps synchronously, correlates results and holds cross-robot dependencies.
102 offline tests pass. A complete PuppyPi Isaac Sim run demonstrated three
successful steps, an absent-object BLOCKED result, and suppression of downstream
work. See DISPATCH_ADAPTER.md for timings and limits. Central world-state ingestion
and feedback-based replanning remain the next integration work.
