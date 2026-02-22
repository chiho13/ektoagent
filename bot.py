import os
import inspect
import asyncio
from dotenv import load_dotenv
from loguru import logger
from typing import Any, Dict, List, Optional

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService
from pipecat.transcriptions.language import Language
from pipecat.frames import frames as pipecat_frames

from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}

BASE_SYSTEM_PROMPT = """
You are Dina, a voice actor companion.
Your purpose is help actor read scripts and improve their voice acting skills.

Speech style rules:
- Insert natural non-verbal vocalizations where appropriate: [sigh], [laugh], [breathe], [clear_throat], [yawn], [cough].
- Use natural spoken fillers occasionally: well, um, so, you know.
- Use contractions and varied sentence length.
- Keep rhythm conversational and voice-friendly.

Think of what emotion you should be expressing based on user's input: [happy], [sad], [angry], [surprised], [fearful], [disgusted]
also consider the delivery style too: [laughing], [whispering]
Output constraints:
- Output only natural spoken sentences.
- Never output markdown, bullet points, headings, or emojis.
""".strip()

LANG_MAP = {
    "en": Language.EN,
    "ja": Language.JA,
    # extend later (es, fr, etc) when your STT supports it and you add mappings
}


def _contains_japanese(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x3040 <= code <= 0x309F  # Hiragana
            or 0x30A0 <= code <= 0x30FF  # Katakana
            or 0x31F0 <= code <= 0x31FF  # Katakana Phonetic Extensions
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0xFF66 <= code <= 0xFF9D  # Halfwidth Katakana
        ):
            return True
    return False


class VoiceRouter(FrameProcessor):
    def __init__(self, tts: InworldTTSService, japanese_voice_id: str, english_voice_id: str):
        super().__init__()
        self._tts = tts
        self._japanese_voice_id = japanese_voice_id
        self._english_voice_id = english_voice_id
        self._current_voice_id: Optional[str] = None
        self._last_input_language: Optional[str] = None

    def _normalize_language(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "value"):
            value = getattr(value, "value")
        text = str(value).strip().lower()
        if not text:
            return None
        if text.startswith("en"):
            return "en"
        if text.startswith("ja"):
            return "ja"
        return text

    async def _set_voice_if_needed(self, voice_id: str):
        if self._current_voice_id == voice_id:
            return
        result = self._tts.set_voice(voice_id)
        if inspect.isawaitable(result):
            await result
        self._current_voice_id = voice_id
        logger.debug("Switched TTS voice to {}", voice_id)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            # Soniox language-identification result on transcription frames.
            frame_lang = self._normalize_language(getattr(frame, "language", None))
            if frame_lang in ("en", "ja"):
                self._last_input_language = frame_lang
                logger.debug("Detected input language from STT: {}", frame_lang)

            # LLM text frames can be TextFrame, LLMTextFrame, etc. Use duck-typing.
            text = getattr(frame, "text", None)
            if isinstance(text, str) and text.strip():
                if self._last_input_language == "en":
                    voice_id = self._japanese_voice_id
                elif self._last_input_language == "ja":
                    voice_id = self._english_voice_id
                else:
                    voice_id = self._japanese_voice_id if _contains_japanese(text) else self._english_voice_id
                await self._set_voice_if_needed(voice_id)

        await self.push_frame(frame, direction)

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def _as_dict(body: Any) -> Dict[str, Any]:
    return body if isinstance(body, dict) else {}

def _parse_language_hints(body: Dict[str, Any]) -> List[Language]:
    # Accept either ["en","ja"] or "en,ja"
    raw = body.get("language_hints") or body.get("languageHints")  # allow both
    if raw is None:
        return [Language.EN]

    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, list):
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
    else:
        raise ValueError("language_hints must be a list like ['en','ja'] or a comma string like 'en,ja'")

    if not parts:
        return [Language.EN]

    out: List[Language] = []
    for p in parts:
        if p not in LANG_MAP:
            raise ValueError(f"Unsupported language hint: {p}")
        out.append(LANG_MAP[p])
    return out

def _parse_bool(body: Dict[str, Any], key1: str, key2: str, default: bool) -> bool:
    v = body.get(key1, body.get(key2, default))
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default

def _compose_system_prompt(body: Dict[str, Any]) -> str:
    # If you allow full override, users can break “translate-only”.
    # Safer: allow addendum only.
    full = body.get("system_prompt") or body.get("systemPrompt")
    addendum = body.get("prompt_addendum") or body.get("promptAddendum")

    if isinstance(full, str) and full.strip():
        return full.strip()

    base = BASE_SYSTEM_PROMPT
    if isinstance(addendum, str) and addendum.strip():
        add = addendum.strip()[:600]
        return base + "\n\nAdditional style preferences:\n" + add

    return base

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    openai_key = _require_env("OPENAI_API_KEY")
    inworld_key = _require_env("INWORLD_API_KEY")
    soniox_key = _require_env("SONIOX_API_KEY")

    body = _as_dict(getattr(runner_args, "body", None))

    # Per-user overrides from requestData
    language_hints = _parse_language_hints(body)
    language_hints_strict = _parse_bool(body, "language_hints_strict", "languageHintsStrict", True)

    voice = body.get("voice") or os.getenv("INWORLD_TTS_VOICE", "Hana")
    japanese_voice = body.get("ja_voice") or body.get("jaVoice") or os.getenv("INWORLD_TTS_VOICE_JA", "Satoshi")
    speaking_rate = float(
        body.get("tts_speed")
        or body.get("ttsSpeed")
        or body.get("speaking_rate")
        or body.get("speakingRate")
        or os.getenv("INWORLD_TTS_SPEAKING_RATE", "1.08")
    )

    system_prompt = _compose_system_prompt(body)
    startup_prompt = (
        body.get("startup_prompt")
        or body.get("startupPrompt")
        or "Start now by greeting yourself as Hana. Get ready to act out a script with me."
    )

    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
    tts_model = os.getenv("INWORLD_TTS_MODEL", "inworld-tts-1.5-max")
    stt_model = os.getenv("SONIOX_STT_MODEL", "stt-rt-v4")

    logger.info(
        "Starting interpreter | hints={} strict={} voice_en={} voice_ja={} speaking_rate={}",
        [h.value for h in language_hints],
        language_hints_strict,
        voice,
        japanese_voice,
        speaking_rate,
    )

    stt = SonioxSTTService(
        api_key=soniox_key,
        params=SonioxInputParams(
            model=stt_model,
            language_hints=language_hints,
            language_hints_strict=language_hints_strict,
            enable_language_identification=True,
        ),
    )

    llm = OpenAILLMService(
        api_key=openai_key,
        model=llm_model,
        params=OpenAILLMService.InputParams(temperature=0.0),
    )

    tts = InworldTTSService(
        api_key=inworld_key,
        model=tts_model,
        voice_id=str(voice),
        params=InworldTTSService.InputParams(
            speaking_rate=speaking_rate,
            auto_mode=True,
        ),
    )
    voice_router = VoiceRouter(
        tts=tts,
        japanese_voice_id=str(japanese_voice),
        english_voice_id=str(voice),
    )

    has_append_frame = hasattr(pipecat_frames, "LLMMessagesAppendFrame")
    has_run_frame = hasattr(pipecat_frames, "LLMRunFrame")

    initial_messages = [{"role": "system", "content": system_prompt}]
    # If append-frame trigger isn't available, seed startup prompt into context.
    if not has_append_frame:
        initial_messages.append({"role": "user", "content": str(startup_prompt)})
    context = LLMContext(initial_messages)

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        voice_router,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    sent_initial_turn = False
    connected_seen = False
    startup_prompt_injected = False
    initial_turn_lock = asyncio.Lock()

    async def _queue_many(frames: List[Any]) -> bool:
        try:
            if hasattr(task, "queue_frames"):
                result = task.queue_frames(frames)
                if inspect.isawaitable(result):
                    await result
            else:
                for frame in frames:
                    result = task.queue_frame(frame)
                    if inspect.isawaitable(result):
                        await result
            return True
        except Exception as e:
            logger.warning("Initial queue attempt failed: {}", e)
            return False

    def _build_initial_trigger_frames() -> tuple[List[Any], bool]:
        if has_append_frame and not startup_prompt_injected:
            return (
                [
                    pipecat_frames.LLMMessagesAppendFrame(
                        [{"role": "user", "content": str(startup_prompt)}],
                        run_llm=True,
                    )
                ],
                True,
            )
        if has_run_frame:
            return ([pipecat_frames.LLMRunFrame()], False)
        return ([user_agg.get_context_frame()], False)

    async def _trigger_initial_turn(reason: str):
        nonlocal sent_initial_turn, startup_prompt_injected
        async with initial_turn_lock:
            if sent_initial_turn:
                return
            logger.info("Queueing initial LLM turn (reason={})", reason)
            frames, marks_prompt_injected = _build_initial_trigger_frames()
            queued = await _queue_many(frames)
            if queued:
                if marks_prompt_injected:
                    startup_prompt_injected = True
                sent_initial_turn = True

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        nonlocal connected_seen
        connected_seen = True
        await _trigger_initial_turn("on_client_connected")

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_transport, _participant):
        nonlocal connected_seen
        connected_seen = True
        await _trigger_initial_turn("on_first_participant_joined")

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(_transport, _participant):
        nonlocal connected_seen
        connected_seen = True
        await _trigger_initial_turn("on_participant_joined")

    @transport.event_handler("on_session_started")
    async def on_session_started(_transport):
        nonlocal connected_seen
        connected_seen = True
        await _trigger_initial_turn("on_session_started")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Client disconnected; cancelling task")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint, handle_sigterm=True)

    async def startup_fallback():
        # Fallback for transports with different connect event timing.
        for delay in (0.5, 1.5, 3.0, 6.0):
            if sent_initial_turn:
                return
            await asyncio.sleep(delay)
            if not connected_seen:
                continue
            await _trigger_initial_turn(f"startup_fallback_{delay}s")

    asyncio.create_task(startup_fallback())
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
