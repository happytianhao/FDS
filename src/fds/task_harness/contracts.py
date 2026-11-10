"""Shared execution rules and compact choices for existing response parameters."""

from fds.common.schema import array, obj, text
from fds.support.protocol import definitions as baseline_definitions
from fds.task_harness.protocol import FRAME_ID
from fds.task_harness.tools import TOOLS


def definitions(budget, gripper, max_pose_targets, max_observation_frames=12):
    tools = baseline_definitions("sys2", budget, None, gripper, max_pose_targets)
    for tool in tools:
        if tool["name"] == "move_ee":
            tool["description"] = (
                "Ordered ABSOLUTE TCP poses in robot base coordinates; quaternion [w,x,y,z], "
                "meters, right-handed axes. Provide the desired reachable target directly, including "
                "long-range moves: there is no configured translation-distance cap or fixed distance "
                "subdivision. The host solves endpoint IK, subdividing rotation or adaptively refining "
                "the path only when IK or joint-displacement constraints require it. Motion uses linear "
                "joint interpolation; the TCP path can curve between endpoints. Integer steps are "
                "shared across support points without changing the total. steps=auto uses joint-motion "
                "requirements, an 8-step movement minimum once per original target, and 4 final "
                "settling steps; intermediate support points have no added hold. Gripper commands take "
                "effect at the START of the original target and persist through its subtargets; "
                "separate an open approach from a stationary close target. All subtargets must pass "
                "IK, joint limits, speed and shared step-budget checks before motion. Feedback "
                "records requested targets and resolved subtargets separately. No collision planner "
                "or contact detection; inspect fresh observations and actual tracking feedback."
            )
    tools.append(
        {
            "name": "review_frames",
            "description": "Inspect or re-inspect selected frames from observation.review_window before taking another environment action. Use its exact window_id and frame IDs (including training: IDs when available). This must be the only tool in an auxiliary batch, with no subgoal or observation_steps fields. No environment steps or physical outcomes are produced. Return to Sys2 with the requested images and the mandatory current final frame. Repeated reviews share the remaining model-call budget. Save supplied keyframes explicitly before the window expires.",
            "arguments": obj(
                window_id=text(80),
                frame_ids=array(FRAME_ID, max_observation_frames - 1),
            ),
        }
    )
    pixel = {**array({"type": "number"}, 2), "minItems": 2}
    frame_id = {"type": "integer", "minimum": 0}
    tools.extend(
        [
            {
                "name": "project_pixels_to_plane",
                "description": "Calculate robot-base coordinates for model-chosen pixels on an explicitly assumed plane n dot p = d. Use already supplied episode frames still in the active review window or explicitly retained keyframes. Each query returns its own result or error, with conditioning and sampled sensitivity to pixel and plane-offset uncertainty. Plane normals must be unit vectors. Uses public per-frame calibration and base_to_world only; cannot identify objects or verify the plane, landing, contact or success. Auxiliary-only batch, with no subgoal or observation_steps fields. No environment steps or extra images; a numerical result does not make old images available for new memory citations.",
                "arguments": obj(
                    queries={
                        **array(
                            obj(
                                frame_id=frame_id,
                                camera=text(80),
                                pixel_xy=pixel,
                                plane_normal_base={**array({"type": "number"}, 3), "minItems": 3},
                                plane_offset_m={"type": "number"},
                                pixel_error_px={"type": "number", "minimum": 0},
                                plane_offset_error_m={"type": "number", "minimum": 0},
                            ),
                            8,
                        ),
                        "minItems": 1,
                    }
                ),
            },
            {
                "name": "check_image_correspondence",
                "description": "Check a model-proposed same-frame cross-camera pixel pair with public camera calibration. Use an already supplied episode frame in the exact active window. Returns epipolar lines and distances in pixels, not an object identity verdict or an acceptance threshold. Different surface points or objects may satisfy the same line. Auxiliary-only batch, with no subgoal or observation_steps fields. No environment steps, extra images or restored memory evidence.",
                "arguments": obj(
                    window_id=text(80),
                    frame_id=frame_id,
                    source_camera=text(80),
                    target_camera=text(80),
                    source_pixel=pixel,
                    target_pixel=pixel,
                ),
            },
        ]
    )
    for tool in tools:
        spec = TOOLS[tool["name"]]
        tool.update(
            kind=spec.kind,
            advances_time=spec.advances_time,
            ends_episode=spec.ends_episode,
            exclusive=spec.exclusive,
        )
    return tools


def execution_contract(limits, *, contract_mode="full"):
    result = {
        "version": "fds_task_execution_v1",
        "response": {
            "tool_batch": "Exactly one batch per response. Environment: {kind: environment, subgoal, tool_calls, observation_steps}. Auxiliary: {kind: auxiliary, tool_calls}; omit subgoal and observation_steps entirely. These batch fields never appear at response top level.",
            "memory_changes": "Unchanged top-level incremental memory updates, in either batch kind.",
            "keyframe_changes": "Unchanged top-level explicit keyframe updates, in either batch kind.",
        },
        "tool_batch.subgoal": "One objective shared by the ordered environment tools. Executed commands do not certify its outcome. Auxiliary batches have no subgoal.",
        "batch": "Each Sys2 decision selects one ordered tool batch, not necessarily an environment advance. Use either environment tools or auxiliary tools in a batch; never mix them. Environment tools run from measured state and share max_tool_steps. A rejected tool, public action error, trial reset or termination cancels later tools; executed prefixes are not rolled back. finish must be last. review_frames is exclusive because it selects the next image set. Independent geometry calls may share an auxiliary batch; read their results before requesting dependent tools in a later Sys2 decision.",
        "tool_kinds": {
            "environment": "Interactions with the environment: wait and move_ee execute steps; finish requests episode termination with zero steps and does not certify success.",
            "auxiliary": "Read-only review or calculations using allowed observations. No environment step and no window expiry. May repeat within the existing model-call budget.",
        },
        "feedback": "last_tool_result is always the previous Sys2 tool batch, including auxiliary results. previous_subgoal is that batch's subgoal for an environment batch, otherwise null. last_environment_result separately preserves the latest batch that actually executed environment steps; it remains historical during auxiliary calls. latest_execution contains raw executed actions only on the next Sys2 input and is never replayed by auxiliary calls.",
        "outcome_checks": {
            "prerequisites": "Before an action that depends on a completed event, check the event's evidence, not just its memory status. An approach, command, image highlight or elapsed time alone does not establish contact, activation or completion. Keep ambiguous transitions unknown and obtain discriminating evidence before the dependent action.",
            "temporal_events": "Sparse demonstration images can show approach, transit, contact and release as similar poses. For ordinal or before/after instructions, review frames on both sides of the relevant transitions before the demo window expires. Record confirmed event order separately from possible events; do not count repeated views of one event twice. If scene items move, track the reference through time into the current scene instead of treating a past image coordinate as an identity.",
            "path_sensitive_actions": "When success depends on the traversed path, endpoint tracking alone does not verify the path between endpoints. Before the demonstration window expires, resolve starts, ends, reversals and transitions from available neighboring frames; distinguish required motion from transit or repeated demonstrations instead of flattening every sampled pose into one continuous route. Preserve these boundaries and unresolved gaps in compact event records and retained keyframes. Sparse samples do not establish interpolation safety: move_ee interpolates joints, so even accurately reached endpoints can produce a different TCP curve. Ground intermediate targets in public evidence. End the batch at an uncertain boundary and compare the executed path with its reference before extending it; observation_steps are delivered after execution and do not interrupt a batch for replanning. If the visible path differs, retain that discrepancy as unresolved evidence and reassess geometry and segment identity before continuing, retracing or extending an irreversible path. Do not treat a visible trace or small endpoint error as semantic success.",
            "count_sensitive_actions": "For exact-count or repetition goals, use existing event memory to track confirmed occurrences separately from unresolved attempts and the resulting possible count range, per required category. Unknown is neither zero nor success. Separate the number of commanded repetitions from the number satisfying the required event relation. Repeatedly matching an earlier endpoint does not independently validate that endpoint: a shared uncertain geometry, direction or event-boundary assumption remains uncertain across all repetitions. Keep the affected count conditional until that shared assumption is checked against the instruction and public reference, rather than promoting it through repetition. Preserve each attempt's evidence and identity uncertainty; a same-looking object appearing after an action is not necessarily the same instance. Before another irreversible action, check whether it could exceed the requested count if earlier unresolved attempts succeeded. For ordered events, also check the next physical transition under both possibilities: the earlier event occurred, or it did not. Leaving and returning to an unresolved endpoint can create a second occurrence or violate order even when the total count is below its limit; calling it inspection, correction or the same attempt does not make it idempotent. Preserve the possible sequence positions in existing event progress and state in the candidate requirements which public evidence makes the proposed return safe under each plausible history. If those histories require incompatible next actions, seek discriminating evidence before replaying or advancing; do not silently select either history. If so, reconcile identities and outcomes using available before/after views or a bounded discriminating observation before adding another occurrence. Do not silently discard uncertain attempts, count repeated views twice, or promote disappearance alone to verified completion.",
            "completion_evidence": "Before finish or any physical action that submits, stops or irreversibly finalizes the task, check each task success criterion separately against current public evidence and any retained demonstration reference. Such a physical action needs completion evidence for the whole prerequisite sequence, not only local contact, alignment or release. Include these prerequisites in the existing selected candidate requirements and required_evidence; do not assume that a later observation can repair an already finalized task. Reproducing the method, causing motion, reaching a nearby region and becoming stationary do not by themselves establish the required final relation. For placement goals, distinguish adjacency or partial image overlap from the demonstrated object-to-target relation; compare relative geometry rather than absolute pixels across different layouts. If a required relation remains ambiguous, keep the event unresolved instead of marking the whole event succeeded with an unverified qualifier. While budget remains, consider an available discriminating review or bounded correction consistent with the task, preserving uncertainty and reserving steps to verify the result. Do not repeat a physical action merely because success is uncertain. Hidden tolerances and official scores must not be guessed. If no supported recovery is available, finish may report inability or unresolved completion honestly; it does not certify success.",
            "identity_continuity": "For a target selected by an earlier cue, keep cue membership, identity across time, and current cross-camera localization as separate claims in existing object/event memory. Compatible projections or epipolar checks constrain current geometry; they do not prove that the current object is the earlier target. When similar objects move, cross, become occluded or remain unattended during other actions, a nearby old position or similar relative arrangement is insufficient to restore identity. Before any environment step, including an open-gripper approach or wait, assess loss of the active review window as a separate consequence from physical contact risk. In the selected candidate requirements, account for unresolved historical associations needed by later stages: resolve their discriminating transitions through review while available, or explicitly retain the supplied evidence needed to revisit them. A physically reversible inspection can irreversibly discard that evidence; closer current views and endpoint keyframes cannot reconstruct an ambiguous earlier crossing. Record remaining gaps and competing associations explicitly instead of silently carrying a stationary assumption forward. Before an irreversible interaction, reconcile the selected object's current identity with the retained cue and intervening motion; do not upgrade tentative identity solely because localization agrees. Use an available discriminating review or bounded observation while uncertainty remains, and end the batch before the identity-dependent action if its evidence must be assessed first.",
            "object_and_robot": "TCP motion is not object motion. Before release, check destination identity and the held object's expected landing relation separately, accounting for object-to-TCP grasp offset, height and perspective across available views. A projected target point is not automatically a TCP destination. Compare the object extent and orientation with the relation required by the instruction or demonstrated placement; a center inside a visible outer boundary does not establish that relation, and the boundary is not an official acceptance tolerance. A conditional projection inherits its pixel identity, surface-height and grasp-offset uncertainties. In existing assumptions and selected candidate requirements, retain unresolved premises behind the landing claim; a successful calculation does not resolve them. Narrowing a query's error range or selecting a different plane on the same image is a revised hypothesis, not new evidence. State what independent visible evidence excludes earlier plausible hypotheses before promoting alignment to supported; otherwise preserve their uncertainty. Compare uncertainty in both object and target estimates with the evidenced landing margin, not just the smaller returned sensitivity. If unresolved, choose an available discriminating review or bounded held reposition that tests a specific premise, retaining its result in existing memory rather than repeating inconclusive probes. Reserve steps for placement and verification. Release may terminate before planned settling observations. Camera-left/right and robot-base axes differ; use supplied task_language_conventions. Successful IK certifies neither identity nor placement.",
            "motion_feedback": "move_ee.target_observations reports measured TCP tracking at each fully commanded original endpoint, not just the final endpoint. Missing endpoints were not observed as fully commanded. These are proprioceptive measurements, not extra images or evidence that an object reached a target. recent_motion_history retains the last six move_ee receipts across auxiliary requests; historical frame IDs do not restore images or memory citation rights.",
            "stalled_progress": "Compare recent_motion_history with observations when progress stalls. Persistent error may reflect contact, grasp offset, orientation or an incorrect reference. Use existing event/experience memory to retain a compact history of failed probes: the hypothesis tested, correction made, observed tracking and object outcome, and what remains unresolved. Withdrawal or a new contract ID does not reset this history. Before another probe, compare it with prior failures and state which unresolved hypothesis it distinguishes and which observable result would change the next decision. A changed target alone is not new evidence. Recomputing geometry on an unverified plane does not resolve feature identity or depth; compare the proposed correction with the returned pixel/plane sensitivity, and retain ambiguity if it cannot be distinguished from that uncertainty. When the same obstruction recurs, reassess the shared reference, grasp offset and orientation using available views before another similar correction; do not accumulate small speculative shifts as if each were independent progress. Choose a bounded discriminating action within the remaining budget and reserve steps to verify its outcome. finish does not certify success and may honestly report inability to complete.",
        },
        "budgets": "Every Sys2 request consumes remaining_calls, including auxiliary-only requests. Only actual environment steps consume the environment and shared batch step budgets. max_tool_calls limits every batch. Episode termination or exhaustion of the episode step/call budget ends the loop; auxiliary tools cannot bypass those limits.",
        "geometry": "Use supplied public calibration and model-specified pixels/assumptions only. check_image_correspondence accepts delivered frames still in the exact active window; project_pixels_to_plane also accepts explicitly retained episode keyframes. Training-video frames have no exposed calibration. Missing, unseen, expired or ill-conditioned data produces errors. Projection returns independent per-query results/errors. Geometry does not return extra images, identify objects, certify outcomes or broaden memory citation permissions.",
        "image_order": "Images and their matching labels are grouped by camera, preserving original within-camera order and episode/training/retained provenance. Pixel values and image selection are unchanged.",
        "move_ee_segments": {
            "auto_split_above_rotation_rad": limits.max_rotation_rad,
            "rule": "There is no translation-distance cap or fixed distance subdivision. Request a reachable destination directly. The host first solves endpoint IK; it subdivides large rotations along the shortest arc and adaptively inserts support poses if IK fails or a joint displacement is too large. The first target starts at measured TCP; subsequent targets start at the previous requested pose. Linear joint interpolation can produce a curved TCP path, not a guaranteed straight Cartesian path. Provide explicit clearance poses when the visible scene requires them.",
            "timing": "Original endpoints and gripper timing are preserved. Integer steps are divided among support points without adding steps, and must allow at least one step per point plus valid joint speed. Auto allocates motion from joint excursions at 80% of the existing per-step limit, with a minimum of 8 movement steps across the whole original target, plus 4 settling steps only at its endpoint. Automatically inserted support points do not add settling. Each explicit original target retains its own timing. Total resolved steps must fit the shared budget.",
            "preflight": "All support poses must pass IK, native joint bounds, joint displacement, per-step speed and shared budget checks before this move_ee starts. Adaptive IK search is bounded by the available step budget; unresolved requests are rejected without action. Subdivision is not collision avoidance.",
            "feedback": "requested_targets preserves the request; resolved_targets and pose_subdivisions include original-target mappings. Tracking refers to the last commanded subtarget; an intermediate subtarget does not mark the original target fully commanded. Observation offsets count actual batch steps.",
        },
        "tool_batch.observation_steps": "Environment batches only. A finish-only batch uses []. Unique positive step offsets over this entire environment-tool batch, bounded by max_tool_steps; at most max_observation_frames-1 requests. The final available frame is always included. Unreached offsets are reported, never invented. review_frames instead addresses existing frame IDs, not future step offsets.",
        "review_frames": {
            "scope": "Initially, the window contains the environment's public cue/demo and initial frames, plus the optional verified training video. After execution, only the most recent executed tool batch is reviewable. Compact inclusive frame_ranges list available IDs; an ID listing is not visual evidence.",
            "lifetime": "Review repeatedly before the next environment step, within remaining_calls. wait and move_ee close the old window immediately before their first actual step, opening one window for the whole new batch. Zero-step preflight rejection preserves the window. finish or episode end closes it. Only explicit keyframes survive expiry; audit and old input files are never retrieval sources.",
            "delivery": "Each review replaces the pending image selection, adds the mandatory final state frame and uses the same per-call observation frame budget. Original IDs, cue flags, training provenance, robot state and acquisition time remain unchanged. Review advances no simulation time and is not a new physical attempt. Raw actions already delivered are not replayed as a new execution.",
            "memory": "Memory deltas and keyframe saves apply to images supplied in this request, before the requested review is delivered next turn. Unseen cached frames cannot yet be evidence or saved keyframes. Training: IDs are demonstration evidence, never current-episode state or execution progress.",
        },
        "choice_semantics": "All choices below remain available for the applicable batch kind each turn. Use existing response fields; labels are descriptions, not new fields or persistent modes. Skills guide the model, never select host modes or override contracts. Planned samples and the mandatory final frame are supplied after execution. Other public batch frames stay in the temporary review window, not automatically in the model input or retained memory.",
        "parameter_choices": {
            "tool_batch.observation_steps": {
                "final_only": "[]",
                "distributed": "Explicit offsets spread across the batch.",
                "targeted": "Explicit offsets near selected transitions; may mix with distributed offsets.",
            },
            "keyframe_changes": {
                "keep": "[] preserves retained frames.",
                "save": "save(frame_id, description, reason)",
                "delete": "delete(frame_id)",
            },
            "review_frames.arguments.frame_ids": {
                "inspect": "Choose existing IDs from observation.review_window.frame_ranges; repeat or refine before acting.",
                "final_only": "[] returns the current final frame without advancing time.",
            },
            "move_ee.arguments.targets[].steps": {
                "auto": "Host resolves bounded IK motion and settling duration.",
                "integer": "Model chooses an explicit duration within the shared budget.",
            },
        },
    }
    if contract_mode != "off":
        result["decision_state"] = {
            "version": "fds_v1",
            "required": "Every response includes fds_contract.",
            "purpose": "Connect the selected tool batch to explicit assumptions, evidence references, expected outcome and falsifiers.",
            "candidate_rule": "List the selected candidate and any plausible alternatives. Every requires ID in every candidate, including unselected alternatives, must name an entry in assumptions. Use tentative or unknown assumption status for unresolved alternatives; never reference an undefined ID. A candidate may be committed only when its required assumptions are supported and required evidence is resolved.",
            "required_evidence_rule": "required_evidence lists prerequisites for the selected batch, not information the batch is intended to obtain. Describe intended observations or calculation outputs in expected_outcome. A committed batch requires every prerequisite resolved from available evidence. If prerequisites remain unresolved, preserve that uncertainty and use a tentative contract for an appropriate low-risk information-gathering batch; do not relabel claims as supported or resolved to pass validation.",
            "evidence_rule": "Cite only frames delivered in this request or explicitly retained keyframes. Missing evidence is unknown, not success.",
            "outcome_rule": "Command execution is not semantic success. Public receipts can contradict a contract; otherwise semantic status remains uncertain until a later decision verifies it.",
            "risk_rule": "High-risk actions require non-empty falsifiers and a committed contract; unresolved assumptions must not be silently treated as facts.",
        }
    return result
