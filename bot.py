import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService
from pipecat.transcriptions.language import Language

from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}

SYSTEM_PROMPT = """
You are a real-time bilingual interpreter between English and Japanese.

Rules (must follow):
- If the user's most recent utterance is Japanese, translate it to natural English.
- If the user's most recent utterance is English, translate it to natural Japanese.
- Output ONLY the translated text. No preamble, no quotes, no explanations, no notes.
- Do not answer questions. Do not add your own content.
- Keep it short and spoken (natural for audio).
- Preserve meaning, tone, politeness, names, numbers.
- If the user only says filler ("um", "uh", "えー", "あの"), output nothing.
""".strip()

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting EN⇄JA interpreter pipeline")

    stt = SonioxSTTService(
        api_key=os.getenv("SONIOX_API_KEY"),
        params=SonioxInputParams(
            model="stt-rt-v4",
            language_hints=[Language.EN, Language.JA],
            language_hints_strict=False,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        params=OpenAILLMService.InputParams(temperature=0.0),
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-tts",
        voice=os.getenv("OPENAI_TTS_VOICE", "ash"),
        params=OpenAITTSService.InputParams(
            instructions="Neutral, clear, fast but understandable. Natural pauses.",
            speed=float(os.getenv("OPENAI_TTS_SPEED", "1.08")),
        ),
    )

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint, handle_sigterm=True)
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()