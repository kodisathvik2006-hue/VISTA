import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  analyzeVideo,
  createInvestigation,
  getCandidates,
  getHealth,
  getInvestigation,
  getInvestigationResults,
  resolveAssetUrl,
  updateEventReview,
} from "./services/api";
import type {
  CandidateDetection,
  CandidatesResponse,
  HealthResponse,
  InvestigationJobDetailResponse,
  InvestigationJobResponse,
  InvestigationMode,
  ReviewStatus,
  VideoAnalyzeResponse,
} from "./types/api";
import { NeurovisionApiError } from "./types/api";

type NavSection = "investigate" | "overview";
type WorkflowStep = "upload" | "select_mode" | "configure" | "results";
type UploadStatus = "idle" | "uploading" | "ready" | "error";

const ACCEPTED = ".mp4,.avi,.mov,.mkv,.webm";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "Unknown";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

export default function App() {
  const [navSection, setNavSection] = useState<NavSection>("investigate");
  const [currentStep, setCurrentStep] = useState<WorkflowStep>("upload");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Video Upload State
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [analyzedVideo, setAnalyzedVideo] = useState<VideoAnalyzeResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Investigation Mode Selection & Config State
  const [selectedMode, setSelectedMode] = useState<InvestigationMode>("specific_target");

  // Mode 2: Specific Target Config State
  const [candidates, setCandidates] = useState<CandidateDetection[]>([]);
  const [candidatesLoading, setCandidatesLoading] = useState(false);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const [targetClass, setTargetClass] = useState<string>("car");
  const [targetColor, setTargetColor] = useState<string>("all");
  const [targetPlate, setTargetPlate] = useState<string>("");
  const [cameraZoneLabel, setCameraZoneLabel] = useState<string>("Monitored Gate Alpha");

  // Mode 1: Suspicious Activity Config State
  const [stopDurationThreshold, setStopDurationThreshold] = useState<number>(4.0);
  const [repeatedCrossingThreshold, setRepeatedCrossingThreshold] = useState<number>(3);

  // Mode 3: Accident Config State
  const [ttcThreshold, setTtcThreshold] = useState<number>(1.5);

  // Mode 4: Altercation Config State
  const [proximityPixels, setProximityPixels] = useState<number>(75);
  const [motionEnergyIndex, setMotionEnergyIndex] = useState<number>(0.5);

  // Common Confidence Threshold
  const [confThreshold, setConfThreshold] = useState<number>(0.25);

  // Active Investigation Job State
  const [currentInvJob, setCurrentInvJob] = useState<InvestigationJobResponse | null>(null);
  const [invResults, setInvResults] = useState<InvestigationJobDetailResponse | null>(null);
  const [invError, setInvError] = useState<string | null>(null);
  const [isLaunchingJob, setIsLaunchingJob] = useState(false);

  // Polling ref for investigation job
  const pollTimerRef = useRef<number | null>(null);
  const videoPlayerRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((data) => {
        if (!cancelled) {
          setHealth(data);
          setHealthError(null);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setHealth(null);
          setHealthError(error instanceof Error ? error.message : "API unreachable");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll active investigation job
  useEffect(() => {
    if (!currentInvJob) return;

    if (currentInvJob.status === "queued" || currentInvJob.status === "processing") {
      pollTimerRef.current = window.setTimeout(async () => {
        try {
          const updated = await getInvestigation(currentInvJob.job_id);
          setCurrentInvJob(updated);
          if (updated.status === "completed") {
            const detail = await getInvestigationResults(updated.job_id);
            setInvResults(detail);
          } else if (updated.status === "failed") {
            setInvError(updated.error ?? "Investigation failed during processing.");
          }
        } catch (err: unknown) {
          setInvError(err instanceof Error ? err.message : "Failed to fetch investigation status.");
        }
      }, 800);
    }

    return () => {
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
      }
    };
  }, [currentInvJob]);

  // Load candidate objects on frame 0 when entering configuration for Mode 2
  useEffect(() => {
    if (analyzedVideo && currentStep === "configure" && selectedMode === "specific_target") {
      setCandidatesLoading(true);
      getCandidates(analyzedVideo.id, 0, confThreshold)
        .then((res: CandidatesResponse) => {
          setCandidates(res.candidates);
          if (res.candidates.length > 0 && selectedCandidateId === null) {
            setSelectedCandidateId(res.candidates[0].candidate_id);
            setTargetClass(res.candidates[0].class_name);
            if (res.candidates[0].dominant_color !== "unspecified") {
              setTargetColor(res.candidates[0].dominant_color);
            }
          }
        })
        .catch(() => {
          // Fall back gracefully if candidates preview cannot be generated
        })
        .finally(() => {
          setCandidatesLoading(false);
        });
    }
  }, [analyzedVideo, currentStep, selectedMode, confThreshold]);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setUploadStatus("uploading");
    setUploadMessage(`Validating container & probing ${file.name}…`);
    setAnalyzedVideo(null);
    setCurrentInvJob(null);
    setInvResults(null);
    setInvError(null);
    setSelectedCandidateId(null);

    try {
      const analyzed = await analyzeVideo(file);
      setAnalyzedVideo(analyzed);
      setUploadStatus("ready");
      setUploadMessage("Video container validated. Ready for Investigation Mode Selection.");
      setCurrentStep("select_mode");
    } catch (error: unknown) {
      setUploadStatus("error");
      if (error instanceof NeurovisionApiError) {
        setUploadMessage(`${error.code}: ${error.message}`);
      } else if (error instanceof Error) {
        setUploadMessage(error.message);
      } else {
        setUploadMessage("Upload failed.");
      }
    }
  }

  function handleSelectMode(mode: InvestigationMode) {
    setSelectedMode(mode);
    setCurrentStep("configure");
  }

  async function handleLaunchInvestigation() {
    if (!analyzedVideo) return;
    setIsLaunchingJob(true);
    setInvError(null);
    setInvResults(null);

    try {
      const selectedCand = candidates.find((c) => c.candidate_id === selectedCandidateId);
      const req = {
        video_id: analyzedVideo.id,
        mode: selectedMode,
        target_config:
          selectedMode === "specific_target"
            ? {
                target_criteria: {
                  candidate_id: selectedCandidateId,
                  target_class: targetClass !== "all" ? targetClass : selectedCand?.class_name,
                  target_color: targetColor !== "all" ? targetColor : selectedCand?.dominant_color,
                  target_plate: targetPlate.trim() ? targetPlate.trim() : null,
                  seed_bbox: selectedCand?.bbox ?? null,
                  seed_frame_index: 0,
                },
                confidence_threshold: confThreshold,
                camera_zone_label: cameraZoneLabel,
              }
            : undefined,
        suspicious_config:
          selectedMode === "suspicious_activity"
            ? {
                stop_duration_threshold_seconds: stopDurationThreshold,
                repeated_crossing_threshold: repeatedCrossingThreshold,
                confidence_threshold: confThreshold,
                classes: ["person", "car", "motorcycle", "truck", "bus"],
              }
            : undefined,
        accident_config:
          selectedMode === "accident_detection"
            ? {
                confidence_threshold: confThreshold,
                ttc_threshold_seconds: ttcThreshold,
              }
            : undefined,
        altercation_config:
          selectedMode === "altercation_detection"
            ? {
                confidence_threshold: confThreshold,
                proximity_threshold_pixels: proximityPixels,
                motion_energy_threshold: motionEnergyIndex,
              }
            : undefined,
        save_annotated_video: true,
      };

      const job = await createInvestigation(req);
      setCurrentInvJob(job);
      setCurrentStep("results");
    } catch (err: unknown) {
      setInvError(err instanceof Error ? err.message : "Failed to launch investigation.");
    } finally {
      setIsLaunchingJob(false);
    }
  }

  async function handleUpdateReview(eventId: string, status: ReviewStatus) {
    if (!currentInvJob || !invResults) return;
    try {
      const updatedEvent = await updateEventReview(currentInvJob.job_id, eventId, status);
      setInvResults({
        ...invResults,
        events: invResults.events.map((e) => (e.event_id === eventId ? updatedEvent : e)),
        summary: invResults.summary
          ? {
              ...invResults.summary,
              review_pending_count: invResults.events.filter(
                (e) => (e.event_id === eventId ? updatedEvent : e).review_status === "Pending Human Review"
              ).length,
            }
          : null,
      });
    } catch (err: unknown) {
      console.error("Failed to update event review status:", err);
    }
  }

  function seekToTimestamp(seconds: number) {
    if (videoPlayerRef.current) {
      videoPlayerRef.current.currentTime = Math.max(0, seconds);
      videoPlayerRef.current.play().catch(() => {});
    }
  }

  function handleFrameImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!analyzedVideo || candidates.length === 0) return;
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const scaleX = analyzedVideo.metadata.width / rect.width;
    const scaleY = analyzedVideo.metadata.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Find candidate containing click point
    const match = candidates.find((c) => {
      const [x1, y1, x2, y2] = c.bbox;
      return clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2;
    });

    if (match) {
      setSelectedCandidateId(match.candidate_id);
      setTargetClass(match.class_name);
      if (match.dominant_color !== "unspecified") {
        setTargetColor(match.dominant_color);
      }
    }
  }

  function handleExportJson() {
    if (!invResults) return;
    const blob = new Blob([JSON.stringify(invResults, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `investigation_report_${invResults.job_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const previewSrc = useMemo(
    () => (analyzedVideo ? resolveAssetUrl(analyzedVideo.preview_url) : null),
    [analyzedVideo]
  );

  return (
    <div className="flex min-h-screen bg-ink-950 text-slate-100 font-sans">
      {/* Sidebar */}
      <aside className="hidden w-72 shrink-0 border-r border-white/10 bg-ink-900/90 p-5 md:flex md:flex-col">
        <div className="mb-6">
          <p className="text-xs uppercase tracking-[0.2em] text-accent-400 font-mono">VISTA Platform</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-white">NEUROVISION</h1>
          <p className="mt-1 text-xs text-slate-400">Investigation-Based Video Intelligence</p>
        </div>

        {/* Workflow Progression Stepper */}
        <div className="mb-6 rounded-xl border border-white/10 bg-ink-800/60 p-4">
          <p className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-3">Investigation Flow</p>
          <nav className="space-y-2 text-xs">
            <StepItem
              number="1"
              label="Video Ingestion"
              active={currentStep === "upload"}
              completed={Boolean(analyzedVideo)}
              onClick={() => setCurrentStep("upload")}
            />
            <StepItem
              number="2"
              label="Investigation Mode"
              active={currentStep === "select_mode"}
              completed={Boolean(analyzedVideo && currentStep !== "upload")}
              disabled={!analyzedVideo}
              onClick={() => analyzedVideo && setCurrentStep("select_mode")}
            />
            <StepItem
              number="3"
              label="Mode Configuration"
              active={currentStep === "configure"}
              completed={Boolean(currentInvJob)}
              disabled={!analyzedVideo}
              onClick={() => analyzedVideo && setCurrentStep("configure")}
            />
            <StepItem
              number="4"
              label="Evidence & Incident Review"
              active={currentStep === "results"}
              completed={invResults?.status === "completed"}
              disabled={!currentInvJob}
              onClick={() => currentInvJob && setCurrentStep("results")}
            />
          </nav>
        </div>

        {/* Navigation buttons */}
        <nav className="space-y-1 text-sm">
          <NavButton active={navSection === "investigate"} onClick={() => setNavSection("investigate")}>
            Investigation Console
          </NavButton>
          <NavButton active={navSection === "overview"} onClick={() => setNavSection("overview")}>
            Architecture & Ethical Bounds
          </NavButton>
        </nav>

        {/* System telemetry note */}
        <div className="mt-auto space-y-2">
          <div className="rounded-xl border border-accent-500/30 bg-accent-500/10 p-3 text-xs text-slate-300">
            <p className="font-semibold text-accent-400">Target Isolation Active:</p>
            VISTA does not blindly assign persistent IDs to every object. Persistent tracking is isolated exclusively to investigator-selected targets.
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 p-4 md:p-8 overflow-y-auto">
        {/* Top Header */}
        <header className="mb-6 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
          <div>
            <h2 className="text-xl font-bold md:text-2xl text-white">
              {navSection === "overview"
                ? "Platform System Architecture & Constraints"
                : currentStep === "upload"
                ? "Step 1: Video Stream Ingestion & Container Probe"
                : currentStep === "select_mode"
                ? "Step 2: Investigation Mode Selection"
                : currentStep === "configure"
                ? "Step 3: Mode-Specific Rule & Target Configuration"
                : "Step 4: Investigation Evidence & Incident Review"}
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              {currentStep === "select_mode"
                ? "Choose an investigation discipline tailored to your security or forensic requirement."
                : currentStep === "configure"
                ? "Define observable event rules or confirm candidate target before launching inference."
                : currentStep === "results"
                ? "Audit flagged incidents, inspect target kinematics, and record investigator review decisions."
                : "OpenCV container-level validation with path-isolated UUID governance."}
            </p>
          </div>
          <HealthBadge health={health} error={healthError} />
        </header>

        {navSection === "overview" ? (
          <OverviewView />
        ) : (
          <div>
            {/* STEP 1: UPLOAD & CONTAINER PROBE */}
            {currentStep === "upload" && (
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                <section className="space-y-6">
                  <UploadPanel
                    dragOver={dragOver}
                    setDragOver={setDragOver}
                    onFile={onFile}
                    disabled={uploadStatus === "uploading"}
                  />
                  <StatusPanel status={uploadStatus} message={uploadMessage} />
                  {analyzedVideo && (
                    <div className="panel p-6 border-accent-500/40 bg-accent-500/10 flex items-center justify-between">
                      <div>
                        <h4 className="font-semibold text-white">Container Readability Verified</h4>
                        <p className="text-xs text-slate-300 mt-1">
                          Video container verified with valid frame-0 preview. Proceed to select your investigation mode.
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setCurrentStep("select_mode")}
                        className="rounded-xl bg-accent-500 px-5 py-2.5 text-sm font-semibold text-ink-950 hover:bg-accent-400 transition shadow-lg"
                      >
                        Select Investigation Mode &rarr;
                      </button>
                    </div>
                  )}
                </section>
                <section className="space-y-6">
                  <MetadataGrid result={analyzedVideo} status={uploadStatus} />
                  <PreviewPanel src={previewSrc} status={uploadStatus} filename={analyzedVideo?.original_filename} />
                </section>
              </div>
            )}

            {/* STEP 2: INVESTIGATION MODE SELECTION */}
            {currentStep === "select_mode" && (
              <div className="space-y-6">
                <div className="rounded-xl border border-white/10 bg-ink-900/60 p-4 flex items-center justify-between">
                  <div className="text-sm">
                    <span className="text-slate-400">Active Video:</span>{" "}
                    <span className="font-semibold text-white">{analyzedVideo?.original_filename}</span>{" "}
                    <span className="text-slate-500">
                      ({analyzedVideo?.metadata.width}×{analyzedVideo?.metadata.height}, {analyzedVideo?.metadata.frame_count} frames)
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setCurrentStep("upload")}
                    className="text-xs text-slate-400 hover:text-white transition"
                  >
                    &larr; Replace Video
                  </button>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  {/* Mode 1 */}
                  <InvestigationModeCard
                    id="suspicious_activity"
                    icon="🛡️"
                    title="1. Detect Suspicious Activity Indicators"
                    subtitle="Zone Perimeter & Observational Security"
                    description="Evaluates observable physical events such as restricted-zone entry, unusual stationary stops in monitored zones, and repeated crossing patterns."
                    ethicalNotice="Strictly observational. Does not classify people as criminals or infer personal suspicion."
                    selected={selectedMode === "suspicious_activity"}
                    onSelect={() => handleSelectMode("suspicious_activity")}
                  />

                  {/* Mode 2 */}
                  <InvestigationModeCard
                    id="specific_target"
                    icon="🎯"
                    title="2. Track a Specific Vehicle or Object"
                    subtitle="Target Isolation & Visual Forensic Following"
                    description="Select a candidate target by clicking its bounding box in a representative video frame or filtering by class and identifiable color. Persistent tracking IDs are restricted solely to the confirmed target."
                    ethicalNotice="Target-only tracking. No facial recognition or biometric identity inference."
                    selected={selectedMode === "specific_target"}
                    onSelect={() => handleSelectMode("specific_target")}
                  />

                  {/* Mode 3 */}
                  <InvestigationModeCard
                    id="accident_detection"
                    icon="💥"
                    title="3. Detect Potential Accidents"
                    subtitle="Kinematic Trajectory & Collision Analysis"
                    description="Evaluates vehicle trajectory convergence, closing velocities, and estimated Time-To-Collision (TTC) to flag potential collision events and near-misses."
                    ethicalNotice="Kinematic indicator only. All results are labeled potential incidents requiring human review."
                    selected={selectedMode === "accident_detection"}
                    onSelect={() => handleSelectMode("accident_detection")}
                  />

                  {/* Mode 4 */}
                  <InvestigationModeCard
                    id="altercation_detection"
                    icon="⚠️"
                    title="4. Detect Possible Physical Altercations"
                    subtitle="Interaction Interface & Action Modeling"
                    description="Evaluates spatial proximity between individuals and erratic kinematic turbulence. Interfaces with temporal action models (VideoMAE / SlowFast) with mandatory human-in-the-loop audit."
                    ethicalNotice="Transparent pipeline: YOLO alone does not detect fights. Preliminary indicators flagged for review."
                    selected={selectedMode === "altercation_detection"}
                    onSelect={() => handleSelectMode("altercation_detection")}
                  />
                </div>
              </div>
            )}

            {/* STEP 3: MODE-SPECIFIC CONFIGURATION */}
            {currentStep === "configure" && analyzedVideo && (
              <div className="space-y-6">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div>
                    <h3 className="text-lg font-semibold text-white">
                      {selectedMode === "specific_target"
                        ? "Configure Target Selection (Specific Target Mode)"
                        : selectedMode === "suspicious_activity"
                        ? "Configure Observational Zone Rules (Suspicious Indicators Mode)"
                        : selectedMode === "accident_detection"
                        ? "Configure Kinematic Thresholds (Potential Accident Mode)"
                        : "Configure Interaction Indicators (Physical Altercation Mode)"}
                    </h3>
                    <p className="text-xs text-slate-400">
                      Fine-tune parameters prior to launching the asynchronous investigation worker.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setCurrentStep("select_mode")}
                    className="text-xs text-slate-400 hover:text-white transition"
                  >
                    &larr; Change Mode
                  </button>
                </div>

                {/* MODE 2: SPECIFIC TARGET CONFIG */}
                {selectedMode === "specific_target" && (
                  <div className="space-y-6">
                    {/* Candidate Object Selector from Frame 0 */}
                    <div className="panel p-6 border-accent-500/30">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="font-semibold text-white">Select Candidate from Representative Frame</h4>
                          <p className="text-xs text-slate-400">
                            Click directly on any vehicle or object bounding box in the frame image below, or choose a candidate badge.
                          </p>
                        </div>
                        {candidatesLoading && (
                          <span className="text-xs text-accent-400 animate-pulse">Detecting candidates on frame 0…</span>
                        )}
                      </div>

                      {/* Interactive Visual Frame for Direct BBox Selection */}
                      {previewSrc && (
                        <div className="mb-4 relative rounded-xl overflow-hidden border border-white/10 bg-black flex justify-center group">
                          <img
                            src={previewSrc}
                            alt="Representative frame for target selection"
                            onClick={handleFrameImageClick}
                            className="max-h-72 w-auto object-contain cursor-crosshair select-none"
                            title="Click directly inside any target bounding box to select"
                          />
                          <div className="absolute top-2 left-2 bg-ink-950/80 border border-white/10 rounded px-2 py-1 text-[11px] text-slate-300 font-mono pointer-events-none">
                            💡 Direct Selection: Click bounding box on frame
                          </div>
                        </div>
                      )}

                      {candidates.length > 0 ? (
                        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                          {candidates.map((c) => {
                            const isSelected = selectedCandidateId === c.candidate_id;
                            return (
                              <button
                                key={c.candidate_id}
                                type="button"
                                onClick={() => {
                                  setSelectedCandidateId(c.candidate_id);
                                  setTargetClass(c.class_name);
                                  if (c.dominant_color !== "unspecified") {
                                    setTargetColor(c.dominant_color);
                                  }
                                }}
                                className={`rounded-xl border p-3 text-left transition ${
                                  isSelected
                                    ? "border-accent-400 bg-accent-500/20 shadow-md ring-1 ring-accent-400"
                                    : "border-white/10 bg-ink-800 hover:border-white/20 hover:bg-white/5"
                                }`}
                              >
                                <div className="flex items-center justify-between text-xs mb-1">
                                  <span className="font-bold text-accent-400 font-mono">Candidate #{c.candidate_id}</span>
                                  <span className="text-slate-400 font-mono">{Math.round(c.confidence * 100)}%</span>
                                </div>
                                <div className="text-sm font-semibold text-white capitalize">{c.class_name}</div>
                                <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-300">
                                  <span className="inline-block h-2.5 w-2.5 rounded-full bg-accent-400"></span>
                                  <span>Color: <strong className="text-white capitalize">{c.dominant_color}</strong></span>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="rounded-xl border border-white/10 bg-ink-800 p-4 text-xs text-slate-400">
                          {candidatesLoading
                            ? "Analyzing frame 0 for objects…"
                            : "No objects auto-detected on frame 0. You can specify the target class and color criteria manually below."}
                        </div>
                      )}
                    </div>

                    {/* Manual Target Filters & Criteria */}
                    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4 panel p-6">
                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">Target Object Class</label>
                        <select
                          value={targetClass}
                          onChange={(e) => setTargetClass(e.target.value)}
                          className="mt-1.5 w-full rounded-lg bg-ink-800 border border-white/10 px-3 py-2 text-sm text-white"
                        >
                          <option value="car">Car</option>
                          <option value="motorcycle">Motorcycle</option>
                          <option value="truck">Truck</option>
                          <option value="bus">Bus</option>
                          <option value="bicycle">Bicycle</option>
                          <option value="person">Person</option>
                        </select>
                      </div>

                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">Vehicle / Object Color</label>
                        <select
                          value={targetColor}
                          onChange={(e) => setTargetColor(e.target.value)}
                          className="mt-1.5 w-full rounded-lg bg-ink-800 border border-white/10 px-3 py-2 text-sm text-white capitalize"
                        >
                          <option value="all">Any Color</option>
                          <option value="black">Black</option>
                          <option value="white">White</option>
                          <option value="silver/gray">Silver / Gray</option>
                          <option value="red">Red</option>
                          <option value="blue">Blue</option>
                          <option value="green">Green</option>
                          <option value="yellow">Yellow</option>
                          <option value="orange">Orange</option>
                        </select>
                      </div>

                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">
                          Registration / Plate OCR (Optional)
                        </label>
                        <input
                          type="text"
                          placeholder="e.g. KA-01-AB-1234"
                          value={targetPlate}
                          onChange={(e) => setTargetPlate(e.target.value)}
                          className="mt-1.5 w-full rounded-lg bg-ink-800 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 font-mono"
                        />
                      </div>

                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">Camera Zone Label</label>
                        <input
                          type="text"
                          value={cameraZoneLabel}
                          onChange={(e) => setCameraZoneLabel(e.target.value)}
                          className="mt-1.5 w-full rounded-lg bg-ink-800 border border-white/10 px-3 py-2 text-sm text-white"
                        />
                      </div>
                    </div>

                    {/* Ethical & Calibration Notice */}
                    <div className="rounded-xl border border-white/10 bg-ink-900 p-4 text-xs text-slate-300 space-y-1">
                      <p className="font-semibold text-accent-400">Target Isolation & Ethical Compliance:</p>
                      <p>
                        • Persistent tracking IDs will be assigned <strong>exclusively to the selected target</strong>.
                        Background objects will not be persistently indexed.
                      </p>
                      <p>
                        • No facial recognition or personal identity inference is used. Real-world street coordinates
                        require extrinsic camera calibration.
                      </p>
                    </div>
                  </div>
                )}

                {/* MODE 1: SUSPICIOUS ACTIVITY CONFIG */}
                {selectedMode === "suspicious_activity" && (
                  <div className="space-y-6">
                    <div className="grid gap-6 sm:grid-cols-2 panel p-6">
                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">
                          Stationary Stop Threshold: <span className="text-accent-400 font-bold">{stopDurationThreshold.toFixed(1)}s</span>
                        </label>
                        <input
                          type="range"
                          min="1.0"
                          max="15.0"
                          step="0.5"
                          value={stopDurationThreshold}
                          onChange={(e) => setStopDurationThreshold(parseFloat(e.target.value))}
                          className="mt-2 w-full accent-accent-500"
                        />
                        <p className="text-xs text-slate-400 mt-1">
                          Flags vehicles or objects remaining stationary within monitored zones exceeding this duration.
                        </p>
                      </div>

                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">
                          Repeated Crossing Threshold: <span className="text-accent-400 font-bold">{repeatedCrossingThreshold} transitions</span>
                        </label>
                        <input
                          type="range"
                          min="2"
                          max="8"
                          step="1"
                          value={repeatedCrossingThreshold}
                          onChange={(e) => setRepeatedCrossingThreshold(parseInt(e.target.value))}
                          className="mt-2 w-full accent-accent-500"
                        />
                        <p className="text-xs text-slate-400 mt-1">
                          Flags multiple back-and-forth crossings between monitored zones within a 30-second window.
                        </p>
                      </div>
                    </div>

                    <div className="rounded-xl border border-white/10 bg-ink-900 p-4 text-xs text-slate-300 space-y-1">
                      <p className="font-semibold text-accent-400">Ethical Framework Compliance:</p>
                      <p>
                        Physical boundary and stopping rules detect observable physical events only. The platform does not
                        classify individuals as criminals or automatically declare persons suspicious.
                      </p>
                    </div>
                  </div>
                )}

                {/* MODE 3: ACCIDENT DETECTION CONFIG */}
                {selectedMode === "accident_detection" && (
                  <div className="space-y-6">
                    <div className="panel p-6">
                      <label className="text-xs uppercase tracking-wide text-slate-400">
                        Time-To-Collision (TTC) Risk Threshold: <span className="text-accent-400 font-bold">{ttcThreshold.toFixed(1)}s</span>
                      </label>
                      <input
                        type="range"
                        min="0.5"
                        max="3.5"
                        step="0.1"
                        value={ttcThreshold}
                        onChange={(e) => setTtcThreshold(parseFloat(e.target.value))}
                        className="mt-2 w-full accent-accent-500 max-w-md"
                      />
                      <p className="text-xs text-slate-400 mt-1">
                        Calculates trajectory convergence and relative closing speeds for converging vehicle pairs.
                      </p>
                    </div>

                    <div className="rounded-xl border border-white/10 bg-ink-900 p-4 text-xs text-slate-300">
                      <p className="font-semibold text-accent-400">Kinematic Verification Protocol:</p>
                      <p className="mt-1">
                        All identified collision courses and bounding-box intersections are marked as preliminary indicators.
                        Mandatory human review is required to verify perspective vs physical impact.
                      </p>
                    </div>
                  </div>
                )}

                {/* MODE 4: PHYSICAL ALTERCATION CONFIG */}
                {selectedMode === "altercation_detection" && (
                  <div className="space-y-6">
                    <div className="grid gap-6 sm:grid-cols-2 panel p-6">
                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">
                          Proximity Threshold: <span className="text-accent-400 font-bold">{proximityPixels}px</span>
                        </label>
                        <input
                          type="range"
                          min="30"
                          max="150"
                          step="5"
                          value={proximityPixels}
                          onChange={(e) => setProximityPixels(parseInt(e.target.value))}
                          className="mt-2 w-full accent-accent-500"
                        />
                        <p className="text-xs text-slate-400 mt-1">
                          Monitors distance between interacting person detections.
                        </p>
                      </div>

                      <div>
                        <label className="text-xs uppercase tracking-wide text-slate-400">
                          Motion Energy Sensitivity: <span className="text-accent-400 font-bold">{Math.round(motionEnergyIndex * 100)}%</span>
                        </label>
                        <input
                          type="range"
                          min="0.2"
                          max="0.9"
                          step="0.05"
                          value={motionEnergyIndex}
                          onChange={(e) => setMotionEnergyIndex(parseFloat(e.target.value))}
                          className="mt-2 w-full accent-accent-500"
                        />
                        <p className="text-xs text-slate-400 mt-1">
                          Evaluates centroid displacement variance for erratic physical movement.
                        </p>
                      </div>
                    </div>

                    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-200">
                      <p className="font-semibold">Action Recognition Pipeline Interface:</p>
                      <p className="mt-1">
                        YOLOv8 alone cannot classify human intent or fight actions. This interface triggers preliminary
                        kinetic alerts and is designed for modular integration with temporal action models (VideoMAE / SlowFast).
                        All occurrences require human verification.
                      </p>
                    </div>
                  </div>
                )}

                {/* Common Launch Bar */}
                <div className="panel p-6 flex flex-wrap items-center justify-between gap-4 border-t border-white/10">
                  <div className="flex items-center gap-4">
                    <div>
                      <span className="text-xs uppercase text-slate-400">Inference Confidence</span>
                      <div className="flex items-center gap-2 mt-1">
                        <input
                          type="range"
                          min="0.10"
                          max="0.80"
                          step="0.05"
                          value={confThreshold}
                          onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                          className="w-28 accent-accent-500"
                        />
                        <span className="text-xs font-mono font-bold text-accent-400">
                          {Math.round(confThreshold * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={handleLaunchInvestigation}
                    disabled={isLaunchingJob}
                    className="rounded-xl bg-accent-500 px-8 py-3 text-sm font-bold text-ink-950 hover:bg-accent-400 disabled:opacity-50 transition shadow-lg"
                  >
                    {isLaunchingJob ? "Launching Investigation…" : "Launch Investigation Worker &rarr;"}
                  </button>
                </div>
              </div>
            )}

            {/* STEP 4: INVESTIGATION RESULTS & REVIEW */}
            {currentStep === "results" && (
              <div className="space-y-6">
                {/* Error Banner */}
                {invError && (
                  <div className="rounded-xl border border-red-400/40 bg-red-500/10 p-4 text-sm text-red-200">
                    <p className="font-semibold">Investigation Job Error</p>
                    <p className="mt-1">{invError}</p>
                  </div>
                )}

                {/* Processing Progress Panel */}
                {currentInvJob && currentInvJob.status !== "completed" && (
                  <div className="panel p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-medium text-white">Investigation Worker Executing</h3>
                        <p className="text-xs text-slate-400">
                          Mode: <span className="font-bold text-accent-400 uppercase">{currentInvJob.mode}</span> | Processing frame by frame…
                        </p>
                      </div>
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-medium uppercase ${
                          currentInvJob.status === "failed"
                            ? "bg-red-500/20 text-red-300"
                            : "bg-blue-500/20 text-blue-300 animate-pulse"
                        }`}
                      >
                        {currentInvJob.status}
                      </span>
                    </div>

                    <div className="mt-4">
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>
                          Frame {currentInvJob.current_frame} / {currentInvJob.total_frames}
                        </span>
                        <span>{Math.round(currentInvJob.progress)}%</span>
                      </div>
                      <div className="w-full bg-ink-800 rounded-full h-2.5 overflow-hidden">
                        <div
                          className="bg-accent-500 h-2.5 rounded-full transition-all duration-300"
                          style={{ width: `${currentInvJob.progress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Completed Investigation View */}
                {invResults && invResults.status === "completed" && (
                  <div className="space-y-6">
                    {/* Mode 2: Target Telemetry HUD (Target Isolation Mode) */}
                    {invResults.summary?.target_telemetry && (
                      <div className="panel p-6 border-accent-500/40 bg-ink-900/80">
                        <div className="flex flex-wrap items-center justify-between border-b border-white/10 pb-4 mb-4 gap-2">
                          <div>
                            <span className="rounded-full bg-accent-500/20 text-accent-300 border border-accent-500/40 px-3 py-0.5 text-xs font-bold uppercase tracking-wide">
                              Isolated Target Telemetry
                            </span>
                            <h3 className="text-xl font-bold text-white mt-2">
                              Target #{invResults.summary.target_telemetry.assigned_track_id ?? "1"}{" "}
                              <span className="capitalize text-accent-400">
                                {invResults.summary.target_telemetry.target_class ?? "Object"}
                              </span>{" "}
                              {invResults.summary.target_telemetry.target_color && (
                                <span className="text-sm font-normal text-slate-300">
                                  ({invResults.summary.target_telemetry.target_color})
                                </span>
                              )}
                            </h3>
                          </div>
                          <div className="text-right text-xs text-slate-400">
                            <div>Monitored Zone: <strong className="text-white">{cameraZoneLabel}</strong></div>
                            <div>Track ID Assignment: <strong className="text-accent-400">Isolated to Target Only</strong></div>
                          </div>
                        </div>

                        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                          <div className="rounded-xl bg-ink-800 p-3">
                            <span className="text-xs text-slate-400 uppercase">First Seen</span>
                            <p className="text-lg font-bold text-white mt-1">
                              {invResults.summary.target_telemetry.first_seen_timestamp != null
                                ? `${invResults.summary.target_telemetry.first_seen_timestamp.toFixed(2)}s`
                                : "N/A"}
                            </p>
                          </div>
                          <div className="rounded-xl bg-ink-800 p-3">
                            <span className="text-xs text-slate-400 uppercase">Last Seen</span>
                            <p className="text-lg font-bold text-white mt-1">
                              {invResults.summary.target_telemetry.last_seen_timestamp != null
                                ? `${invResults.summary.target_telemetry.last_seen_timestamp.toFixed(2)}s`
                                : "N/A"}
                            </p>
                          </div>
                          <div className="rounded-xl bg-ink-800 p-3">
                            <span className="text-xs text-slate-400 uppercase">Total Visible Duration</span>
                            <p className="text-lg font-bold text-accent-400 mt-1">
                              {invResults.summary.target_telemetry.total_visible_duration_seconds.toFixed(2)}s
                            </p>
                          </div>
                          <div className="rounded-xl bg-ink-800 p-3">
                            <span className="text-xs text-slate-400 uppercase">Avg Tracking Confidence</span>
                            <p className="text-lg font-bold text-white mt-1">
                              {Math.round(invResults.summary.target_telemetry.average_confidence * 100)}%
                            </p>
                          </div>
                        </div>

                        {/* Stopping & Lost Visibility Breakdown */}
                        <div className="mt-4 grid gap-4 md:grid-cols-2 text-xs">
                          <div className="rounded-xl bg-ink-800/60 p-3 border border-white/5">
                            <span className="font-semibold text-slate-300">Observed Stopping Intervals:</span>
                            {invResults.summary.target_telemetry.observed_stopping_intervals.length > 0 ? (
                              <ul className="mt-2 space-y-1">
                                {invResults.summary.target_telemetry.observed_stopping_intervals.map((s, idx) => (
                                  <li key={idx} className="text-slate-400">
                                    • Stop {idx + 1}: {s.start.toFixed(1)}s &rarr; {s.end.toFixed(1)}s ({s.duration.toFixed(1)}s stationary)
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="mt-1 text-slate-500">Continuous movement observed; no prolonged stops.</p>
                            )}
                          </div>

                          <div className="rounded-xl bg-ink-800/60 p-3 border border-white/5">
                            <span className="font-semibold text-slate-300">Lost Visibility / Occlusion Intervals:</span>
                            {invResults.summary.target_telemetry.lost_visibility_intervals.length > 0 ? (
                              <ul className="mt-2 space-y-1">
                                {invResults.summary.target_telemetry.lost_visibility_intervals.map((o, idx) => (
                                  <li key={idx} className="text-slate-400">
                                    • Occlusion: {o.start.toFixed(1)}s &rarr; {o.end.toFixed(1)}s ({o.duration.toFixed(1)}s unobserved)
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="mt-1 text-slate-500">Unbroken optical line of sight throughout capture.</p>
                            )}
                          </div>
                        </div>

                        <p className="text-[11px] text-slate-400 mt-3 italic">
                          *{invResults.summary.target_telemetry.calibration_note}
                        </p>
                      </div>
                    )}

                    {/* Main Results Grid: Video Player + Incident Timeline */}
                    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                      {/* Video Player */}
                      <div className="panel overflow-hidden">
                        <div className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
                          <div>
                            <h3 className="text-lg font-medium text-white">Investigation Evidence Video</h3>
                            <p className="text-xs text-slate-400">
                              Annotated with mode-specific indicators, bounding boxes, and telemetry.
                            </p>
                          </div>
                          {invResults.annotated_video_url && (
                            <a
                              href={resolveAssetUrl(invResults.annotated_video_url)}
                              download={`investigation_${invResults.job_id}.mp4`}
                              className="rounded-lg bg-ink-800 border border-white/10 px-3 py-1.5 text-xs text-slate-300 hover:text-white hover:bg-white/5 transition"
                            >
                              Download Video MP4
                            </a>
                          )}
                        </div>
                        <div className="bg-black/60 p-4 flex justify-center">
                          {invResults.annotated_video_url ? (
                            <video
                              ref={videoPlayerRef}
                              src={resolveAssetUrl(invResults.annotated_video_url)}
                              controls
                              playsInline
                              className="max-h-[460px] w-full rounded-lg object-contain bg-black"
                            />
                          ) : (
                            <div className="py-24 text-center text-xs text-slate-500">
                              Annotated video stream not available.
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Flagged Incidents & Events with Human Review Controls */}
                      <div className="panel p-6 flex flex-col h-[560px]">
                        <div className="border-b border-white/10 pb-3 flex items-center justify-between">
                          <div>
                            <h3 className="text-lg font-medium text-white">Flagged Incidents & Events</h3>
                            <p className="text-xs text-slate-400">
                              {invResults.events.length} observable event{invResults.events.length === 1 ? "" : "s"} logged.
                            </p>
                          </div>
                          <span className="rounded-full bg-accent-500/10 text-accent-400 px-3 py-1 text-xs font-semibold">
                            {invResults.summary?.review_pending_count ?? 0} Pending Review
                          </span>
                        </div>

                        {/* Event List */}
                        <div className="flex-1 overflow-y-auto mt-4 space-y-3 pr-1">
                          {invResults.events.length === 0 ? (
                            <div className="text-center py-16 text-slate-400 text-xs">
                              No incidents matching the configured criteria were flagged in this stream.
                            </div>
                          ) : (
                            invResults.events.map((ev) => (
                              <div
                                key={ev.event_id}
                                className="rounded-xl bg-ink-800 border border-white/10 p-3 hover:border-accent-500/40 transition space-y-2"
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <button
                                      type="button"
                                      onClick={() => seekToTimestamp(ev.start_time)}
                                      className="rounded-md bg-accent-500/20 border border-accent-500/40 px-2 py-0.5 text-xs font-mono font-semibold text-accent-300 hover:bg-accent-500/30 transition flex items-center gap-1"
                                      title="Click to jump video to incident timestamp"
                                    >
                                      ▶ {ev.start_time.toFixed(2)}s (F#{ev.frame_index})
                                    </button>
                                    {ev.clip_url && (
                                      <a
                                        href={resolveAssetUrl(ev.clip_url)}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="rounded-md bg-ink-700/80 border border-white/10 px-2 py-0.5 text-[11px] font-mono text-slate-300 hover:text-white hover:bg-white/10 transition"
                                        title="Download/Open focused incident snippet clip"
                                      >
                                        🎬 Clip
                                      </a>
                                    )}
                                  </div>
                                  <ReviewBadge status={ev.review_status} />
                                </div>

                                <div>
                                  <h4 className="text-xs font-bold text-white">{ev.label}</h4>
                                  <p className="text-xs text-slate-300 mt-0.5">{ev.description}</p>
                                </div>

                                <div className="text-[11px] text-slate-400 bg-ink-900/60 rounded p-1.5 italic">
                                  *{ev.uncertainty_note}
                                </div>

                                {/* Review Actions */}
                                <div className="flex items-center justify-between border-t border-white/5 pt-2 text-xs">
                                  <span className="text-slate-400">Investigator Decision:</span>
                                  <div className="flex items-center gap-1.5">
                                    <button
                                      type="button"
                                      onClick={() => handleUpdateReview(ev.event_id, "Verified Incident")}
                                      className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                                        ev.review_status === "Verified Incident"
                                          ? "bg-emerald-500 text-ink-950 font-bold"
                                          : "bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
                                      }`}
                                    >
                                      Verify
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => handleUpdateReview(ev.event_id, "False Positive - Dismissed")}
                                      className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                                        ev.review_status === "False Positive - Dismissed"
                                          ? "bg-red-500 text-white font-bold"
                                          : "bg-red-500/10 text-red-300 hover:bg-red-500/20"
                                      }`}
                                    >
                                      Dismiss
                                    </button>
                                  </div>
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Evidence Frames Gallery */}
                    {invResults.evidence_frame_urls && invResults.evidence_frame_urls.length > 0 && (
                      <div className="panel p-6">
                        <h3 className="text-lg font-medium text-white mb-1">Key Evidence Snapshots</h3>
                        <p className="text-xs text-slate-400 mb-4">
                          Extracted high-quality evidence keyframes captured at critical incident timestamps.
                        </p>
                        <div className="grid gap-3 grid-cols-2 sm:grid-cols-4 lg:grid-cols-6">
                          {invResults.evidence_frame_urls.map((url, idx) => (
                            <a
                              key={idx}
                              href={resolveAssetUrl(url)}
                              target="_blank"
                              rel="noreferrer"
                              className="group block overflow-hidden rounded-lg border border-white/10 hover:border-accent-400 transition"
                            >
                              <img
                                src={resolveAssetUrl(url)}
                                alt={`Evidence ${idx + 1}`}
                                className="h-24 w-full object-cover group-hover:scale-105 transition"
                              />
                              <div className="bg-ink-800 p-1.5 text-[10px] text-center text-slate-300 font-mono">
                                Evidence #{idx + 1}
                              </div>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Audit Summary & Pipeline Notes */}
                    <div className="panel p-6">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs uppercase tracking-wide text-slate-400">Forensic Audit & Compliance Record</h4>
                        <button
                          type="button"
                          onClick={handleExportJson}
                          className="rounded-lg bg-accent-500/15 border border-accent-500/30 px-3 py-1.5 text-xs text-accent-300 hover:bg-accent-500/25 transition font-semibold"
                        >
                          📥 Export Forensic Report (JSON)
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-4 text-xs text-slate-300">
                        <div>Processing Speed: <strong className="text-white">{invResults.summary?.fps_processed} FPS</strong></div>
                        <div>Total Frames Processed: <strong className="text-white">{invResults.summary?.total_frames_processed}</strong></div>
                        <div>Job Duration: <strong className="text-white">{invResults.summary?.processing_time_seconds}s</strong></div>
                      </div>
                      <div className="mt-3 space-y-1">
                        {invResults.summary?.notes.map((n, i) => (
                          <p key={i} className="text-xs text-slate-400">• {n}</p>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

// -------------------------------------------------------------------------
// Sub-components
// -------------------------------------------------------------------------

function StepItem({
  number,
  label,
  active,
  completed,
  disabled,
  onClick,
}: {
  number: string;
  label: string;
  active: boolean;
  completed?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`w-full flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left transition ${
        active
          ? "bg-accent-500/20 text-accent-400 font-bold"
          : completed
          ? "text-slate-200 hover:bg-white/5"
          : disabled
          ? "opacity-40 cursor-not-allowed text-slate-500"
          : "text-slate-400 hover:bg-white/5"
      }`}
    >
      <span
        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
          active
            ? "bg-accent-500 text-ink-950"
            : completed
            ? "bg-accent-500/20 text-accent-400 border border-accent-500/40"
            : "bg-ink-700 text-slate-400"
        }`}
      >
        {completed && !active ? "✓" : number}
      </span>
      <span className="truncate">{label}</span>
    </button>
  );
}

function InvestigationModeCard({
  icon,
  title,
  subtitle,
  description,
  ethicalNotice,
  selected,
  onSelect,
}: {
  id?: string;
  icon: string;
  title: string;
  subtitle: string;
  description: string;
  ethicalNotice: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={`cursor-pointer rounded-2xl border p-6 transition flex flex-col justify-between ${
        selected
          ? "border-accent-400 bg-accent-500/10 shadow-xl ring-2 ring-accent-400"
          : "border-white/10 bg-ink-900/80 hover:border-accent-500/40 hover:bg-ink-800/80"
      }`}
    >
      <div>
        <div className="flex items-center justify-between">
          <span className="text-3xl">{icon}</span>
          <span className="rounded-full bg-white/5 border border-white/10 px-2.5 py-0.5 text-[11px] text-slate-400 uppercase font-mono">
            {subtitle}
          </span>
        </div>
        <h3 className="mt-4 text-lg font-bold text-white">{title}</h3>
        <p className="mt-2 text-xs text-slate-300 leading-relaxed">{description}</p>
      </div>

      <div className="mt-5 pt-4 border-t border-white/5">
        <p className="text-[11px] text-accent-400/90 italic">*{ethicalNotice}</p>
        <button
          type="button"
          className={`mt-4 w-full rounded-xl py-2.5 text-xs font-bold transition shadow ${
            selected
              ? "bg-accent-500 text-ink-950"
              : "bg-ink-800 text-white hover:bg-white/10 border border-white/10"
          }`}
        >
          {selected ? "Selected Mode &rarr;" : "Select This Investigation Mode"}
        </button>
      </div>
    </div>
  );
}

function ReviewBadge({ status }: { status: ReviewStatus }) {
  if (status === "Verified Incident") {
    return (
      <span className="rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 px-2.5 py-0.5 text-[10px] font-semibold">
        Verified Incident
      </span>
    );
  }
  if (status === "False Positive - Dismissed") {
    return (
      <span className="rounded-full bg-red-500/15 border border-red-500/30 text-red-300 px-2.5 py-0.5 text-[10px] font-semibold">
        Dismissed (False Positive)
      </span>
    );
  }
  return (
    <span className="rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 px-2.5 py-0.5 text-[10px] font-semibold">
      Pending Human Review
    </span>
  );
}

function NavButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full rounded-lg px-3 py-2 text-left transition ${
        active ? "bg-accent-500/15 text-accent-400 font-medium" : "text-slate-300 hover:bg-white/5"
      }`}
    >
      {children}
    </button>
  );
}

function HealthBadge({ health, error }: { health: HealthResponse | null; error: string | null }) {
  if (error) {
    return (
      <div className="rounded-full border border-red-400/40 bg-red-500/10 px-3 py-1 text-xs text-red-200">
        API offline — {error}
      </div>
    );
  }
  if (!health) {
    return (
      <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
        Checking API…
      </div>
    );
  }
  return (
    <div className="rounded-full border border-accent-500/30 bg-accent-500/10 px-3 py-1 text-xs text-accent-400 font-mono">
      {health.service} v{health.version} · {health.device?.toUpperCase()} · YOLO & ByteTrack {health.ai_tracking ? "Ready" : "Offline"}
    </div>
  );
}

function OverviewView() {
  return (
    <div className="space-y-6">
      <div className="panel p-6 border-accent-500/30 bg-accent-500/5">
        <span className="rounded-full bg-accent-500/20 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-accent-300">
          Platform Governance
        </span>
        <h3 className="mt-3 text-lg font-semibold text-white">
          Investigation-Based Video Intelligence Architecture
        </h3>
        <p className="mt-2 text-sm text-slate-300 leading-relaxed">
          VISTA transforms video surveillance from generic, noisy tracking of every visible object into an
          investigation-directed workflow. Persistent tracking is isolated exclusively to confirmed targets, and
          incident flags operate with strict human-in-the-loop review.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="panel p-5 space-y-3">
          <h4 className="font-semibold text-accent-400 text-sm">Target Isolation & Privacy Governance</h4>
          <ul className="text-xs text-slate-300 space-y-2">
            <li>• Persistent tracking IDs are restricted exclusively to investigator-selected targets.</li>
            <li>• No facial recognition or biometric identity classification is performed.</li>
            <li>• All physical speed and coordinate telemetry are clearly designated as uncalibrated estimates unless extrinsic camera calibration is configured.</li>
          </ul>
        </div>

        <div className="panel p-5 space-y-3">
          <h4 className="font-semibold text-accent-400 text-sm">Action & Collision Integrity</h4>
          <ul className="text-xs text-slate-300 space-y-2">
            <li>• YOLO neural models identify objects, not human intent or complex altercation actions.</li>
            <li>• Altercation indicators evaluate spatial proximity and motion energy, with modular slots for temporal action models (VideoMAE / SlowFast).</li>
            <li>• All flagged incidents require investigator verification (Verified vs Dismissed False Positive).</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function UploadPanel({
  dragOver,
  setDragOver,
  onFile,
  disabled,
}: {
  dragOver: boolean;
  setDragOver: (value: boolean) => void;
  onFile: (file: File | undefined) => void;
  disabled: boolean;
}) {
  return (
    <div className="panel p-6">
      <h3 className="text-lg font-medium text-white">Video Stream Ingestion</h3>
      <p className="mt-1 text-xs text-slate-400">
        Upload standard traffic or surveillance stream (.mp4, .avi, .mov, .mkv, .webm up to 200MB).
      </p>
      <label
        onDragOver={(event) => {
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          onFile(event.dataTransfer.files[0]);
        }}
        className={`mt-4 block cursor-pointer rounded-2xl border-2 border-dashed px-6 py-12 text-center transition ${
          dragOver ? "border-accent-400 bg-accent-500/10" : "border-white/15 bg-ink-800/60"
        } ${disabled ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          type="file"
          accept={ACCEPTED}
          className="hidden"
          disabled={disabled}
          onChange={(event) => onFile(event.target.files?.[0])}
        />
        <p className="text-base font-medium text-white">Drop video here or click to browse</p>
        <p className="mt-2 text-xs text-slate-400">Validated directly at the binary OpenCV container level.</p>
      </label>
    </div>
  );
}

function StatusPanel({ status, message }: { status: UploadStatus; message: string | null }) {
  const label =
    status === "idle"
      ? "Awaiting File"
      : status === "uploading"
      ? "Streaming & Probing"
      : status === "ready"
      ? "Validated"
      : "Error";
  return (
    <div className="panel p-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">Container Status</h3>
        <span
          className={`rounded-full px-3 py-1 text-xs ${
            status === "error"
              ? "bg-red-500/15 text-red-200"
              : status === "ready"
              ? "bg-accent-500/15 text-accent-400"
              : "bg-white/10 text-slate-300"
          }`}
        >
          {label}
        </span>
      </div>
      <p className="mt-3 text-xs text-slate-300">{message ?? "Upload a video file to begin investigation."}</p>
    </div>
  );
}

function MetadataGrid({ result, status }: { result: VideoAnalyzeResponse | null; status: UploadStatus }) {
  if (status === "uploading") {
    return (
      <div className="panel p-6">
        <h3 className="text-sm font-medium text-white">Metadata</h3>
        <p className="mt-3 animate-pulse text-xs text-slate-400">Probing FPS, duration, and frame dimensions…</p>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="panel p-6">
        <h3 className="text-sm font-medium text-white">Metadata</h3>
        <p className="mt-3 text-xs text-slate-400">
          Awaiting video upload to extract duration, resolution, FPS, and frame count.
        </p>
      </div>
    );
  }
  const items = [
    ["File", result.original_filename],
    ["Size", formatBytes(result.size_bytes)],
    ["Duration", formatDuration(result.metadata.duration_seconds)],
    ["FPS", String(result.metadata.fps)],
    ["Frames", String(result.metadata.frame_count)],
    ["Resolution", `${result.metadata.width} × ${result.metadata.height}`],
  ];
  return (
    <div className="panel p-6">
      <h3 className="text-sm font-medium text-white">Container Metadata</h3>
      <dl className="mt-4 grid grid-cols-2 gap-3">
        {items.map(([label, value]) => (
          <div key={label} className="rounded-xl bg-ink-800 px-3 py-3">
            <dt className="text-[11px] uppercase tracking-wide text-slate-400">{label}</dt>
            <dd className="mt-1 truncate text-xs text-white font-mono" title={value}>
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function PreviewPanel({
  src,
  status,
  filename,
}: {
  src: string | null;
  status: UploadStatus;
  filename?: string;
}) {
  return (
    <div className="panel overflow-hidden">
      <div className="border-b border-white/10 px-6 py-4">
        <h3 className="text-sm font-medium text-white">Representative Frame (Frame 0)</h3>
        <p className="text-xs text-slate-400">Path-isolated JPEG preview served via UUID governance.</p>
      </div>
      <div className="flex min-h-56 items-center justify-center bg-black/40 p-4">
        {status === "uploading" ? (
          <p className="text-xs text-slate-400 animate-pulse">Extracting preview…</p>
        ) : src ? (
          <img src={src} alt={`Preview of ${filename ?? "video"}`} className="max-h-80 rounded-lg object-contain" />
        ) : (
          <p className="text-xs text-slate-400">No preview available.</p>
        )}
      </div>
    </div>
  );
}
