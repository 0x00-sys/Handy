use crate::actions::ACTION_MAP;
use crate::managers::audio::AudioRecordingManager;
use log::{debug, error, warn};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

const DEBOUNCE: Duration = Duration::from_millis(30);
const RELEASE_GRACE: Duration = Duration::from_millis(50);

/// The longest hold still counted as a "tap" when looking for a double-tap.
/// A real push-to-talk hold runs longer than this, so ordinary dictation never
/// pays the `DOUBLE_TAP_WINDOW` wait below.
const TAP_MAX_HOLD: Duration = Duration::from_millis(250);

/// How long a tap's stop is deferred while we wait to see whether a second tap
/// arrives to latch hands-free.
const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(300);

/// A re-press sooner than this after a release is key chatter or X11
/// auto-repeat, never a human double-tap.
const DOUBLE_TAP_MIN_GAP: Duration = Duration::from_millis(40);

/// How long the shortcut must be held before the overlay offers the hands-free
/// key: long enough that ordinary dictation never sees the hint. Once the user
/// has latched hands-free at least once the hint is never shown again.
const HANDS_FREE_HINT_DELAY: Duration = Duration::from_millis(10_000);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PttAction {
    Passthrough,
    /// Hold the stop back briefly. `is_tap` marks a release short enough to be
    /// the first half of a double-tap, which earns the longer window.
    DeferRelease {
        is_tap: bool,
    },
    CancelRelease,
    /// Second tap of a double-tap: keep recording, hands-free.
    Latch,
}

struct PendingRelease {
    binding_id: String,
    hotkey_string: String,
    deadline: Instant,
    /// Set when the release ended a tap rather than a hold, so a re-press
    /// inside the window latches instead of merely cancelling the stop.
    is_tap: bool,
    released_at: Instant,
}

/// How long a deferred release is held back before its stop fires.
fn defer_window(is_tap: bool) -> Duration {
    if is_tap {
        DOUBLE_TAP_WINDOW
    } else {
        RELEASE_GRACE
    }
}

/// The subset of `PendingRelease` that classification needs.
#[derive(Debug, Clone, Copy)]
struct PendingReleaseView<'a> {
    binding_id: &'a str,
    is_tap: bool,
    /// Time since the release was deferred.
    since_release: Duration,
}

/// Everything `classify_ptt_event` reads, gathered so the state machine stays
/// a pure function of its inputs and can be unit-tested without a running app.
#[derive(Debug, Clone, Copy)]
struct PttInput<'a> {
    pending_release: Option<PendingReleaseView<'a>>,
    is_pressed: bool,
    push_to_talk: bool,
    /// The `push_to_talk_hands_free` setting.
    hands_free: bool,
    binding_id: &'a str,
    /// `(binding currently recording, whether it is already latched)`.
    recording: Option<(&'a str, bool)>,
    /// How long the key has been down, measured at this event. Auto-repeat
    /// presses do not restart it — it measures the whole hold.
    hold_elapsed: Duration,
}

/// Commands processed sequentially by the coordinator thread.
enum Command {
    Input {
        binding_id: String,
        hotkey_string: String,
        is_pressed: bool,
        push_to_talk: bool,
        hands_free: bool,
    },
    /// The dedicated hands-free key fired during a push-to-talk hold.
    HandsFreeLatch,
    Cancel {
        recording_was_active: bool,
    },
    ProcessingFinished,
}

/// Pipeline lifecycle, owned exclusively by the coordinator thread.
enum Stage {
    Idle,
    Recording {
        binding_id: String,
        /// Hands-free: the shortcut is no longer holding the recording open,
        /// so releases are ignored and the next press stops it.
        latched: bool,
        /// Post-process the transcript when this recording ends. Set when the
        /// post-process binding started the recording, or when it fired while
        /// another binding was already recording — its combo may share its
        /// leading keys with the transcribe binding, which then always wins
        /// the initial press (e.g. `option` vs `option+shift+space`).
        post_process: bool,
    },
    Processing,
}

impl Stage {
    fn recording(&self) -> Option<(&str, bool)> {
        match self {
            Stage::Recording {
                binding_id,
                latched,
                ..
            } => Some((binding_id.as_str(), *latched)),
            _ => None,
        }
    }
}

fn classify_ptt_event(input: PttInput) -> PttAction {
    if !input.push_to_talk {
        return PttAction::Passthrough;
    }

    if input.is_pressed {
        match input.pending_release {
            // The key came back down before the deferred stop fired. A tap
            // re-pressed inside the double-tap window is the user asking for
            // hands-free; anything else is auto-repeat holding the key open.
            // The window is bounded here rather than by the timer alone: a busy
            // coordinator can dequeue a press after its deadline, and a late
            // press must never latch (and drop) the stop it was racing.
            Some(pending) if pending.binding_id == input.binding_id => {
                let is_second_tap = input.hands_free
                    && pending.is_tap
                    && (DOUBLE_TAP_MIN_GAP..DOUBLE_TAP_WINDOW).contains(&pending.since_release)
                    // Without an unlatched recording there is nothing to latch,
                    // and latching would silently discard the deferred stop.
                    && input
                        .recording
                        .is_some_and(|(id, latched)| id == input.binding_id && !latched);
                if is_second_tap {
                    PttAction::Latch
                } else {
                    PttAction::CancelRelease
                }
            }
            _ => PttAction::Passthrough,
        }
    } else {
        match input.recording {
            // Every release is deferred, latched or not. Unlatched, the delay
            // absorbs auto-repeat before the stop fires; latched, nothing fires
            // on expiry, but the deferral still lets an auto-repeat press be
            // recognised and swallowed instead of ending the recording.
            Some((id, latched)) if id == input.binding_id && input.pending_release.is_none() => {
                PttAction::DeferRelease {
                    is_tap: input.hands_free && !latched && input.hold_elapsed <= TAP_MAX_HOLD,
                }
            }
            _ => PttAction::Passthrough,
        }
    }
}

/// Fire a deferred release whose window has closed. Called from the timer arm,
/// and again before any input is handled: a coordinator that was busy past a
/// deadline dequeues the next event with `Ok` rather than `Timeout`, and it must
/// still behave exactly like one that woke on time.
fn expire_pending_release(
    app: &AppHandle,
    stage: &mut Stage,
    pending_release: &mut Option<PendingRelease>,
    hint_deadline: &mut Option<Instant>,
    hold_started: &mut Option<Instant>,
    now: Instant,
) {
    let Some(pending) = pending_release.take_if(|pending| pending.deadline <= now) else {
        return;
    };
    // A latched recording has no stop to fire here: the deferral existed only
    // to absorb auto-repeat.
    if stops_recording(stage, &pending.binding_id, false) {
        *hint_deadline = None;
        *hold_started = None;
        stop(app, stage, &pending.binding_id, &pending.hotkey_string);
    }
}

/// Whether this event ends the recording: a latched one waits for the next
/// press, an ordinary hold ends when the key it started with comes back up.
fn stops_recording(stage: &Stage, binding_id: &str, is_pressed: bool) -> bool {
    match stage {
        Stage::Recording {
            binding_id: id,
            latched,
            ..
        } => id == binding_id && *latched == is_pressed,
        _ => false,
    }
}

/// A press of one transcribe binding while the other holds the recording open.
/// The two combos may share their leading keys (`option` for push-to-talk and
/// `option+shift+space` for post-processing), so the shorter one always wins
/// the initial press and the longer one can only ever fire mid-recording.
/// Retarget the in-flight recording to the incoming binding's mode instead of
/// ignoring the press. Returns whether the press was consumed.
fn switch_transcribe_mode(stage: &mut Stage, incoming: &str) -> bool {
    let Stage::Recording {
        binding_id,
        post_process,
        ..
    } = stage
    else {
        return false;
    };
    if binding_id == incoming {
        return false;
    }
    let want = incoming == "transcribe_with_post_process";
    if *post_process != want {
        *post_process = want;
        debug!(
            "Recording held by '{binding_id}' retargeted by '{incoming}' (post_process: {want})"
        );
    }
    true
}

/// Serialises all transcription lifecycle events through a single thread
/// to eliminate race conditions between keyboard shortcuts, signals, and
/// the async transcribe-paste pipeline.
pub struct TranscriptionCoordinator {
    tx: Sender<Command>,
}

pub fn is_transcribe_binding(id: &str) -> bool {
    id == "transcribe" || id == "transcribe_with_post_process"
}

impl TranscriptionCoordinator {
    pub fn new(app: AppHandle) -> Self {
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut stage = Stage::Idle;
                let mut last_press: Option<Instant> = None;
                let mut pending_release: Option<PendingRelease> = None;
                // Start of the current hold. Auto-repeat presses never reset it,
                // so it measures how long the key has actually been down.
                let mut hold_started: Option<Instant> = None;
                // When to offer the hands-free key in the overlay.
                let mut hint_deadline: Option<Instant> = None;

                loop {
                    let next_deadline = [
                        pending_release.as_ref().map(|pending| pending.deadline),
                        hint_deadline,
                    ]
                    .into_iter()
                    .flatten()
                    .min();

                    let cmd = if let Some(deadline) = next_deadline {
                        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                            Ok(cmd) => cmd,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                let now = Instant::now();

                                if hint_deadline.is_some_and(|at| at <= now) {
                                    hint_deadline = None;
                                    // Once the user has latched hands-free they
                                    // know the feature exists — stop hinting.
                                    if matches!(stage.recording(), Some((_, false)))
                                        && !crate::settings::get_settings(&app).hands_free_used
                                    {
                                        if let Some(key) = crate::shortcut::active_hands_free_key()
                                        {
                                            crate::overlay::emit_hands_free_hint(&app, &key);
                                        }
                                    }
                                }

                                expire_pending_release(
                                    &app,
                                    &mut stage,
                                    &mut pending_release,
                                    &mut hint_deadline,
                                    &mut hold_started,
                                    now,
                                );
                                continue;
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    } else {
                        match rx.recv() {
                            Ok(cmd) => cmd,
                            Err(_) => break,
                        }
                    };

                    match cmd {
                        Command::Input {
                            binding_id,
                            hotkey_string,
                            is_pressed,
                            push_to_talk,
                            hands_free,
                        } => {
                            let now = Instant::now();
                            // Settle any deadline this event overtook, so a late
                            // press is classified against the same state it would
                            // have met had the timer fired first.
                            expire_pending_release(
                                &app,
                                &mut stage,
                                &mut pending_release,
                                &mut hint_deadline,
                                &mut hold_started,
                                now,
                            );

                            let action = classify_ptt_event(PttInput {
                                pending_release: pending_release.as_ref().map(|pending| {
                                    PendingReleaseView {
                                        binding_id: pending.binding_id.as_str(),
                                        is_tap: pending.is_tap,
                                        since_release: now.duration_since(pending.released_at),
                                    }
                                }),
                                is_pressed,
                                push_to_talk,
                                hands_free,
                                binding_id: &binding_id,
                                recording: stage.recording(),
                                hold_elapsed: hold_started
                                    .map(|at| now.duration_since(at))
                                    .unwrap_or(Duration::MAX),
                            });

                            match action {
                                PttAction::Latch => {
                                    if latch(&app, &mut stage) {
                                        pending_release = None;
                                    }
                                    continue;
                                }
                                PttAction::CancelRelease => {
                                    pending_release = None;
                                    continue;
                                }
                                PttAction::DeferRelease { is_tap } => {
                                    pending_release = Some(PendingRelease {
                                        binding_id,
                                        hotkey_string,
                                        deadline: now + defer_window(is_tap),
                                        is_tap,
                                        released_at: now,
                                    });
                                    continue;
                                }
                                PttAction::Passthrough => {}
                            }

                            // Debounce rapid-fire press events (key repeat / double-tap).
                            // Push-to-talk releases may be deferred above to absorb X11 auto-repeat.
                            if is_pressed {
                                if last_press.is_some_and(|t| now.duration_since(t) < DEBOUNCE) {
                                    debug!("Debounced press for '{binding_id}'");
                                    continue;
                                }
                                last_press = Some(now);
                                hold_started = Some(now);
                            }

                            if push_to_talk {
                                if is_pressed && matches!(stage, Stage::Idle) {
                                    start(&app, &mut stage, &binding_id, &hotkey_string);
                                    // The hands-free key is claimed here rather
                                    // than in `TranscribeAction::start`, which
                                    // cannot tell a genuine hold from a
                                    // CLI/signal toggle reusing the same path.
                                    hint_deadline =
                                        if hands_free && matches!(stage, Stage::Recording { .. }) {
                                            crate::shortcut::register_hands_free_shortcut(
                                                &app,
                                                &binding_id,
                                            );
                                            Some(now + HANDS_FREE_HINT_DELAY)
                                        } else {
                                            None
                                        };
                                } else if stops_recording(&stage, &binding_id, is_pressed) {
                                    hint_deadline = None;
                                    hold_started = None;
                                    stop(&app, &mut stage, &binding_id, &hotkey_string);
                                } else if is_pressed {
                                    // The other transcribe binding fired while a
                                    // recording is in flight (its combo shares
                                    // keys with the one holding it): retarget
                                    // the recording instead of dropping it.
                                    switch_transcribe_mode(&mut stage, &binding_id);
                                }
                            } else if is_pressed {
                                match &mut stage {
                                    Stage::Idle => {
                                        start(&app, &mut stage, &binding_id, &hotkey_string);
                                    }
                                    Stage::Recording { binding_id: id, .. }
                                        if id == &binding_id =>
                                    {
                                        hint_deadline = None;
                                        hold_started = None;
                                        stop(&app, &mut stage, &binding_id, &hotkey_string);
                                    }
                                    Stage::Recording { .. } => {
                                        switch_transcribe_mode(&mut stage, &binding_id);
                                    }
                                    _ => {
                                        debug!("Ignoring press for '{binding_id}': pipeline busy")
                                    }
                                }
                            }
                        }
                        Command::HandsFreeLatch => {
                            // Only drop the deferred stop if there was actually a
                            // recording to latch; otherwise it still has to fire.
                            if latch(&app, &mut stage) {
                                pending_release = None;
                            }
                        }
                        Command::Cancel {
                            recording_was_active,
                        } => {
                            pending_release = None;
                            hint_deadline = None;
                            hold_started = None;
                            // Don't reset during processing — wait for the pipeline to finish.
                            if !matches!(stage, Stage::Processing)
                                && (recording_was_active
                                    || matches!(stage, Stage::Recording { .. }))
                            {
                                stage = Stage::Idle;
                            }
                        }
                        Command::ProcessingFinished => {
                            stage = Stage::Idle;
                        }
                    }
                }
                debug!("Transcription coordinator exited");
            }));
            if let Err(e) = result {
                error!("Transcription coordinator panicked: {e:?}");
            }
        });

        Self { tx }
    }

    /// Send a keyboard/signal input event for a transcribe binding.
    /// For signal-based toggles, use `is_pressed: true` and `push_to_talk: false`
    /// (which also makes `hands_free` irrelevant — there is no hold to latch).
    pub fn send_input(
        &self,
        binding_id: &str,
        hotkey_string: &str,
        is_pressed: bool,
        push_to_talk: bool,
        hands_free: bool,
    ) {
        if self
            .tx
            .send(Command::Input {
                binding_id: binding_id.to_string(),
                hotkey_string: hotkey_string.to_string(),
                is_pressed,
                push_to_talk,
                hands_free,
            })
            .is_err()
        {
            warn!("Transcription coordinator channel closed");
        }
    }

    /// Convert an in-flight push-to-talk hold into a hands-free recording.
    pub fn notify_hands_free_latch(&self) {
        if self.tx.send(Command::HandsFreeLatch).is_err() {
            warn!("Transcription coordinator channel closed");
        }
    }

    pub fn notify_cancel(&self, recording_was_active: bool) {
        if self
            .tx
            .send(Command::Cancel {
                recording_was_active,
            })
            .is_err()
        {
            warn!("Transcription coordinator channel closed");
        }
    }

    pub fn notify_processing_finished(&self) {
        if self.tx.send(Command::ProcessingFinished).is_err() {
            warn!("Transcription coordinator channel closed");
        }
    }
}

fn start(app: &AppHandle, stage: &mut Stage, binding_id: &str, hotkey_string: &str) {
    let Some(action) = ACTION_MAP.get(binding_id) else {
        warn!("No action in ACTION_MAP for '{binding_id}'");
        return;
    };
    action.start(app, binding_id, hotkey_string);
    if app
        .try_state::<Arc<AudioRecordingManager>>()
        .is_some_and(|a| a.is_recording())
    {
        *stage = Stage::Recording {
            binding_id: binding_id.to_string(),
            latched: false,
            post_process: binding_id == "transcribe_with_post_process",
        };
    } else {
        debug!("Start for '{binding_id}' did not begin recording; staying idle");
    }
}

fn stop(app: &AppHandle, stage: &mut Stage, binding_id: &str, hotkey_string: &str) {
    // The action decides whether the transcript is post-processed; the audio
    // manager matches on the binding that started the recording, so the
    // original binding_id is still what gets passed to the action.
    let action_id = match stage {
        Stage::Recording {
            post_process: true, ..
        } => "transcribe_with_post_process",
        _ => binding_id,
    };
    let Some(action) = ACTION_MAP.get(action_id) else {
        warn!("No action in ACTION_MAP for '{action_id}'");
        return;
    };
    action.stop(app, binding_id, hotkey_string);
    *stage = Stage::Processing;
}

/// Switch the in-flight hold to hands-free: the shortcut stops holding the
/// recording open and the next press ends it. The hands-free key is released
/// back to the foreground app now that it has done its job.
///
/// Returns whether anything latched — `false` when there was no unlatched
/// recording, so callers know the deferred stop still has to fire.
fn latch(app: &AppHandle, stage: &mut Stage) -> bool {
    let Stage::Recording { latched, .. } = stage else {
        return false;
    };
    if *latched {
        return false;
    }
    *latched = true;
    debug!("Push-to-talk hold latched into hands-free recording");
    crate::shortcut::unregister_hands_free_shortcut(app);
    crate::overlay::emit_hands_free_active(app);
    // First real use: the overlay hint has done its job, never show it again.
    let mut settings = crate::settings::get_settings(app);
    if !settings.hands_free_used {
        settings.hands_free_used = true;
        crate::settings::write_settings(app, settings);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline input: push-to-talk on, hands-free off, nothing pending.
    fn input(is_pressed: bool) -> PttInput<'static> {
        PttInput {
            pending_release: None,
            is_pressed,
            push_to_talk: true,
            hands_free: false,
            binding_id: BINDING,
            recording: None,
            hold_elapsed: Duration::MAX,
        }
    }

    /// A deferred release, re-pressed after a human-length gap.
    fn pending(is_tap: bool) -> PendingReleaseView<'static> {
        PendingReleaseView {
            binding_id: BINDING,
            is_tap,
            since_release: Duration::from_millis(120),
        }
    }

    #[test]
    fn push_to_talk_release_while_recording_defers_release() {
        assert_eq!(
            classify_ptt_event(PttInput {
                recording: Some((BINDING, false)),
                ..input(false)
            }),
            PttAction::DeferRelease { is_tap: false }
        );
    }

    #[test]
    fn push_to_talk_press_matching_pending_release_cancels_release() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(false)),
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::CancelRelease
        );
    }

    #[test]
    fn toggle_mode_press_and_release_pass_through() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(false)),
                push_to_talk: false,
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::Passthrough
        );
        assert_eq!(
            classify_ptt_event(PttInput {
                push_to_talk: false,
                recording: Some((BINDING, false)),
                ..input(false)
            }),
            PttAction::Passthrough
        );
    }

    #[test]
    fn press_for_different_binding_than_pending_release_passes_through() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(false)),
                binding_id: "transcribe_with_post_process",
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::Passthrough
        );
    }

    #[test]
    fn press_matching_pending_release_cancels_without_recording_state() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(false)),
                ..input(true)
            }),
            PttAction::CancelRelease
        );
    }

    // ---------------------------------------------------------------------
    // Transcribe-mode switching (overlapping bindings, e.g. `option` holding
    // the recording while `option+shift+space` fires mid-hold)
    // ---------------------------------------------------------------------

    const POST_PROCESS_BINDING: &str = "transcribe_with_post_process";

    fn recording(binding_id: &str, post_process: bool) -> Stage {
        Stage::Recording {
            binding_id: binding_id.to_string(),
            latched: false,
            post_process,
        }
    }

    fn stage_post_process(stage: &Stage) -> Option<bool> {
        match stage {
            Stage::Recording { post_process, .. } => Some(*post_process),
            _ => None,
        }
    }

    /// The post-process combo fired while the plain transcribe binding holds
    /// the recording: the recording is retargeted, not ignored.
    #[test]
    fn post_process_press_upgrades_an_in_flight_transcribe_recording() {
        let mut stage = recording(BINDING, false);
        assert!(switch_transcribe_mode(&mut stage, POST_PROCESS_BINDING));
        assert_eq!(stage_post_process(&stage), Some(true));
        // The owner of the hold is unchanged, so its release still stops it.
        assert!(stops_recording(&stage, BINDING, false));
    }

    /// Symmetric overlap: transcribe firing during a post-process hold drops
    /// back to a plain transcription.
    #[test]
    fn transcribe_press_downgrades_an_in_flight_post_process_recording() {
        let mut stage = recording(POST_PROCESS_BINDING, true);
        assert!(switch_transcribe_mode(&mut stage, BINDING));
        assert_eq!(stage_post_process(&stage), Some(false));
    }

    /// A press of the binding already holding the recording is not a switch —
    /// it must keep flowing to the normal stop/latch handling.
    #[test]
    fn same_binding_press_is_not_a_mode_switch() {
        let mut stage = recording(BINDING, false);
        assert!(!switch_transcribe_mode(&mut stage, BINDING));
        assert_eq!(stage_post_process(&stage), Some(false));

        let mut stage = Stage::Idle;
        assert!(!switch_transcribe_mode(&mut stage, POST_PROCESS_BINDING));
    }

    // ---------------------------------------------------------------------
    // Hands-free latching
    // ---------------------------------------------------------------------

    /// A short hold released with hands-free enabled is a double-tap candidate,
    /// so its stop earns the longer window instead of the auto-repeat grace.
    #[test]
    fn short_hold_release_is_a_tap_when_hands_free_is_enabled() {
        assert_eq!(
            classify_ptt_event(PttInput {
                hands_free: true,
                recording: Some((BINDING, false)),
                hold_elapsed: Duration::from_millis(120),
                ..input(false)
            }),
            PttAction::DeferRelease { is_tap: true }
        );
    }

    /// The latency guarantee for ordinary dictation: a hold long enough to say
    /// anything is never a tap, so its release keeps the 50 ms grace and stops
    /// as promptly as it always did.
    #[test]
    fn long_hold_release_is_not_a_tap() {
        assert_eq!(
            classify_ptt_event(PttInput {
                hands_free: true,
                recording: Some((BINDING, false)),
                hold_elapsed: TAP_MAX_HOLD + Duration::from_millis(1),
                ..input(false)
            }),
            PttAction::DeferRelease { is_tap: false }
        );
    }

    /// With the setting off, a short hold behaves exactly as before.
    #[test]
    fn short_hold_is_not_a_tap_when_hands_free_is_disabled() {
        assert_eq!(
            classify_ptt_event(PttInput {
                recording: Some((BINDING, false)),
                hold_elapsed: Duration::from_millis(120),
                ..input(false)
            }),
            PttAction::DeferRelease { is_tap: false }
        );
    }

    #[test]
    fn second_tap_inside_the_window_latches() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(true)),
                hands_free: true,
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::Latch
        );
    }

    /// Key chatter and X11 auto-repeat re-press almost instantly; only a human
    /// gap counts as a deliberate second tap.
    #[test]
    fn re_press_inside_the_minimum_gap_does_not_latch() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(PendingReleaseView {
                    since_release: DOUBLE_TAP_MIN_GAP - Duration::from_millis(1),
                    ..pending(true)
                }),
                hands_free: true,
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::CancelRelease
        );
    }

    /// The double-tap window is bounded on both sides by the classifier, not
    /// just by the timer: a busy coordinator thread can dequeue a press after
    /// its deadline, and a late press must not latch — it would also discard the
    /// stop it was racing.
    #[test]
    fn re_press_after_the_window_closed_does_not_latch() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(PendingReleaseView {
                    since_release: DOUBLE_TAP_WINDOW,
                    ..pending(true)
                }),
                hands_free: true,
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::CancelRelease
        );
    }

    /// Latching needs a recording to latch. Without one the deferred stop is
    /// merely cancelled — never silently dropped by a latch that no-ops.
    #[test]
    fn second_tap_without_a_recording_does_not_latch() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(true)),
                hands_free: true,
                ..input(true)
            }),
            PttAction::CancelRelease
        );
    }

    /// Latching requires the deferred release to have been a tap; re-pressing
    /// after a long hold just keeps the existing recording open.
    #[test]
    fn re_press_after_a_long_hold_does_not_latch() {
        assert_eq!(
            classify_ptt_event(PttInput {
                pending_release: Some(pending(false)),
                hands_free: true,
                recording: Some((BINDING, false)),
                ..input(true)
            }),
            PttAction::CancelRelease
        );
    }

    /// Once latched the key no longer holds the recording open, so letting go
    /// must not stop it — including the release of the second tap itself. The
    /// release is still deferred (nothing fires on expiry) so that an
    /// auto-repeat press behind it is recognised rather than read as a stop.
    #[test]
    fn release_while_latched_defers_without_arming_a_tap() {
        assert_eq!(
            classify_ptt_event(PttInput {
                hands_free: true,
                recording: Some((BINDING, true)),
                hold_elapsed: Duration::from_millis(80),
                ..input(false)
            }),
            PttAction::DeferRelease { is_tap: false }
        );
    }

    /// A press while latched is the user ending the hands-free recording; it
    /// passes through to the stop branch rather than being swallowed.
    #[test]
    fn press_while_latched_passes_through_to_stop() {
        assert_eq!(
            classify_ptt_event(PttInput {
                hands_free: true,
                recording: Some((BINDING, true)),
                ..input(true)
            }),
            PttAction::Passthrough
        );
    }

    // ---------------------------------------------------------------------
    // Sequence-level regression coverage for issue #1539.
    //
    // Under X11 key auto-repeat, holding a push-to-talk key does not emit one
    // long press. It emits the initial press followed by a stream of
    // synthesized release/press pairs, then a single genuine release on key-up.
    // Before the fix, every synthesized release passed straight through and
    // stopped recording, so holding the key "rapidly toggled" recording on and
    // off. The fix defers each release for a short grace window and cancels it
    // when the matching auto-repeat press arrives.
    //
    // The unit tests above assert `classify_ptt_event` in isolation. The
    // simulator below threads that classifier through the same `pending_release`
    // / `stage` state transitions the coordinator loop performs (lines that
    // handle `Command::Input` and the `recv_timeout` grace expiry), so a whole
    // event burst can be exercised deterministically without a Tauri AppHandle
    // or real timers.
    // ---------------------------------------------------------------------

    const BINDING: &str = "transcribe";

    #[derive(Clone, Copy)]
    enum Ev {
        /// A key-down event (real initial press or a synthesized auto-repeat press).
        Press,
        /// A key-up event (synthesized auto-repeat release or the genuine key-up).
        Release,
        /// The deferred-release window elapsed with no cancelling press arriving.
        Grace,
        /// Time passing with no key activity, in milliseconds.
        Wait(u64),
    }

    #[derive(Debug, PartialEq, Eq)]
    enum SimStage {
        Idle,
        Recording { latched: bool },
        Processing,
    }

    struct SimResult {
        starts: u32,
        stops: u32,
        stage: SimStage,
    }

    /// Mirror of the coordinator loop's decision logic for a single push-to-talk
    /// binding: it calls the real `classify_ptt_event` and applies the exact same
    /// Defer / Cancel / Latch / debounce / start / stop transitions.
    fn simulate(events: &[Ev], hands_free: bool) -> SimResult {
        let mut stage = SimStage::Idle;
        let mut pending: Option<(bool, u64)> = None; // (is_tap, released_at_ms)
        let mut last_press_ms: Option<u64> = None;
        let mut hold_started_ms: Option<u64> = None;
        let mut clock_ms: u64 = 0;
        let mut starts = 0u32;
        let mut stops = 0u32;
        let debounce_ms = DEBOUNCE.as_millis() as u64;

        for ev in events {
            // Auto-repeat events arrive a few ms apart, well inside DEBOUNCE.
            clock_ms += match ev {
                Ev::Wait(ms) => *ms,
                // Long enough to close either deferral window.
                Ev::Grace => DOUBLE_TAP_WINDOW.as_millis() as u64 + 1,
                _ => 5,
            };

            // Coordinator's `RecvTimeoutError::Timeout` arm: once a deferred
            // release's window elapses it fires, stopping the recording only if
            // the binding is still holding it open.
            if let Some((is_tap, released_at)) = pending {
                if clock_ms - released_at >= defer_window(is_tap).as_millis() as u64 {
                    pending = None;
                    if stage == (SimStage::Recording { latched: false }) {
                        stage = SimStage::Processing;
                        hold_started_ms = None;
                        stops += 1;
                    }
                }
            }

            match ev {
                Ev::Wait(_) | Ev::Grace => {}
                Ev::Press | Ev::Release => {
                    let is_pressed = matches!(ev, Ev::Press);
                    let recording = match &stage {
                        SimStage::Recording { latched } => Some((BINDING, *latched)),
                        _ => None,
                    };

                    let action = classify_ptt_event(PttInput {
                        pending_release: pending.map(|(is_tap, at)| PendingReleaseView {
                            binding_id: BINDING,
                            is_tap,
                            since_release: Duration::from_millis(clock_ms - at),
                        }),
                        is_pressed,
                        push_to_talk: true,
                        hands_free,
                        binding_id: BINDING,
                        recording,
                        hold_elapsed: hold_started_ms
                            .map(|at| Duration::from_millis(clock_ms - at))
                            .unwrap_or(Duration::MAX),
                    });

                    match action {
                        PttAction::Latch => {
                            pending = None;
                            if let SimStage::Recording { latched } = &mut stage {
                                *latched = true;
                            }
                            continue;
                        }
                        PttAction::CancelRelease => {
                            pending = None;
                            continue;
                        }
                        PttAction::DeferRelease { is_tap } => {
                            pending = Some((is_tap, clock_ms));
                            continue;
                        }
                        PttAction::Passthrough => {}
                    }

                    if is_pressed {
                        if last_press_ms.is_some_and(|t| clock_ms - t < debounce_ms) {
                            continue;
                        }
                        last_press_ms = Some(clock_ms);
                        hold_started_ms = Some(clock_ms);
                    }

                    match (&stage, is_pressed) {
                        (SimStage::Idle, true) => {
                            stage = SimStage::Recording { latched: false };
                            starts += 1;
                        }
                        (SimStage::Recording { latched: true }, true)
                        | (SimStage::Recording { latched: false }, false) => {
                            stage = SimStage::Processing;
                            hold_started_ms = None;
                            stops += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        SimResult {
            starts,
            stops,
            stage,
        }
    }

    /// Initial press plus several synthesized release/press pairs, as X11 emits
    /// while a push-to-talk key is held down. The first auto-repeat only arrives
    /// after the system's repeat delay, so the hold is already well past
    /// `TAP_MAX_HOLD` by then.
    fn autorepeat_burst() -> Vec<Ev> {
        let mut events = vec![Ev::Press, Ev::Wait(660)];
        for _ in 0..6 {
            events.push(Ev::Release);
            events.push(Ev::Press);
            events.push(Ev::Wait(40));
        }
        events
    }

    /// Regression for #1539: a burst of X11 auto-repeat release/press pairs must
    /// not stop recording. Before the fix the first synthesized release stopped
    /// recording immediately (stops == 1, stage left Recording), which produced
    /// the rapid on/off toggling. With the fix the releases are coalesced and
    /// recording stays continuously active for the whole burst.
    #[test]
    fn x11_autorepeat_burst_does_not_toggle_recording() {
        let result = simulate(&autorepeat_burst(), false);
        assert_eq!(result.starts, 1, "recording should start exactly once");
        assert_eq!(
            result.stops, 0,
            "synthesized auto-repeat releases must not stop recording mid-burst"
        );
        assert_eq!(
            result.stage,
            SimStage::Recording { latched: false },
            "recording must remain active across the entire auto-repeat burst"
        );
    }

    /// The same burst with hands-free enabled must still not latch: auto-repeat
    /// re-presses arrive far inside `DOUBLE_TAP_MIN_GAP`, and the hold they
    /// interrupt is long past `TAP_MAX_HOLD`.
    #[test]
    fn x11_autorepeat_burst_does_not_latch_hands_free() {
        let result = simulate(&autorepeat_burst(), true);
        assert_eq!(result.starts, 1);
        assert_eq!(result.stops, 0);
        assert_eq!(
            result.stage,
            SimStage::Recording { latched: false },
            "auto-repeat must never be mistaken for a double-tap"
        );
    }

    /// Complements the burst test: once the key is genuinely released and the
    /// grace window elapses with no re-press, recording stops exactly once. This
    /// proves the debounce only coalesces synthesized releases and does not wedge
    /// the coordinator or swallow the real key-up.
    #[test]
    fn genuine_release_after_grace_stops_recording_once() {
        let mut events = autorepeat_burst();
        events.push(Ev::Release); // genuine key-up
        events.push(Ev::Grace); // grace window elapses, no cancelling press
        let result = simulate(&events, false);
        assert_eq!(result.starts, 1, "recording should start exactly once");
        assert_eq!(
            result.stops, 1,
            "a genuine release should stop recording exactly once"
        );
        assert_eq!(result.stage, SimStage::Processing);
    }

    /// The headline flow: tap, tap again inside the window, then speak freely.
    /// Releasing the second tap must not stop anything, and the third press
    /// ends the recording.
    #[test]
    fn double_tap_latches_and_next_press_stops() {
        let events = [
            Ev::Press,
            Ev::Wait(90),
            Ev::Release, // first tap ends -> deferred as a tap
            Ev::Wait(120),
            Ev::Press, // second tap inside the window -> latch
            Ev::Wait(60),
            Ev::Release,     // released while latched -> ignored
            Ev::Wait(8_000), // hands-free dictation
            Ev::Press,       // deliberate stop
            Ev::Wait(80),
            Ev::Release,
        ];
        let result = simulate(&events, true);
        assert_eq!(result.starts, 1, "recording should start exactly once");
        assert_eq!(
            result.stops, 1,
            "only the press after latching should stop the recording"
        );
        assert_eq!(result.stage, SimStage::Processing);
    }

    /// With the setting off, the identical key sequence is two ordinary
    /// push-to-talk taps — proof the feature is inert until enabled.
    #[test]
    fn double_tap_does_not_latch_when_hands_free_is_disabled() {
        let events = [
            Ev::Press,
            Ev::Wait(90),
            Ev::Release,
            Ev::Wait(120), // 50 ms grace elapses -> the first tap stops
        ];
        let result = simulate(&events, false);
        assert_eq!(result.starts, 1);
        assert_eq!(result.stops, 1);
        assert_eq!(result.stage, SimStage::Processing);
    }

    /// A lone tap with hands-free on must still transcribe once the window
    /// closes — the latch path must not strand a recording that never got a
    /// second tap.
    #[test]
    fn single_tap_still_stops_after_the_double_tap_window() {
        let events = [Ev::Press, Ev::Wait(90), Ev::Release, Ev::Wait(300)];
        let result = simulate(&events, true);
        assert_eq!(result.starts, 1);
        assert_eq!(result.stops, 1);
        assert_eq!(result.stage, SimStage::Processing);
    }

    /// A single tap whose second press arrives after the window closed — the
    /// shape a descheduled coordinator produces, since it then dequeues the
    /// press with `Ok` instead of waking on the deadline. The tap must still
    /// transcribe, exactly as if the timer had fired on time.
    #[test]
    fn tap_still_stops_when_a_late_press_overtakes_the_deadline() {
        let events = [
            Ev::Press,
            Ev::Wait(90),
            Ev::Release,
            Ev::Wait(310), // past DOUBLE_TAP_WINDOW: the stop is now due
            Ev::Press,     // dequeued late, must not latch or swallow the stop
        ];
        let result = simulate(&events, true);
        assert_eq!(result.starts, 1);
        assert_eq!(
            result.stops, 1,
            "a late press must not discard the tap's pending stop"
        );
        assert_eq!(result.stage, SimStage::Processing);
    }

    /// The user latches by double-tapping but keeps the key held down, so X11
    /// starts auto-repeating it. Those synthesized presses must not be read as
    /// the deliberate press that ends a hands-free recording — otherwise
    /// latching on X11 would stop the recording a moment later.
    #[test]
    fn autorepeat_while_latched_does_not_stop_recording() {
        let mut events = vec![
            Ev::Press,
            Ev::Wait(90),
            Ev::Release,
            Ev::Wait(120),
            Ev::Press,     // latches, and the key stays down
            Ev::Wait(660), // auto-repeat delay
        ];
        for _ in 0..6 {
            events.push(Ev::Release);
            events.push(Ev::Press);
            events.push(Ev::Wait(40));
        }
        let result = simulate(&events, true);
        assert_eq!(result.starts, 1);
        assert_eq!(
            result.stops, 0,
            "auto-repeat must not end a hands-free recording"
        );
        assert_eq!(result.stage, SimStage::Recording { latched: true });
    }
}
