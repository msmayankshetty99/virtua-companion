"""Discord-specific transport endpoints over the one existing session/model."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...audio.tts_http import synthesize_wav
from ...kernel.cancellation import TurnBusy, TurnCancelled

MAX_AUDIO_SECONDS = 60
MAX_PCM_BYTES = 16000 * 2 * MAX_AUDIO_SECONDS


def transcribe_pcm(session, pcm):
    """A Discord voice message or call segment, through the session's one Whisper model (the microphone's)."""
    return session.transcribe(pcm, beam_size=1, vad_filter=True, condition_on_previous_text=False)


# The Discord process sees only what bot.py acts on: its own turns' text, reasoning and tool approvals, approval and
# Discord resources, initiative messages and whiteboard updates. Local transcripts, chats and desktop state stay local,
# and voice levels or snapshots never fill its socket's queue (closed with 1013) during a long local turn.
CLIENT_EVENTS = frozenset({'resource.discord', 'resource.approvals', 'tool.approval_finished', 'tool.approval_resolved',
    'initiative.presented', 'whiteboard.changed', 'whiteboard.image'})
TURN_EVENTS = frozenset({'chat.delta', 'model.reasoning', 'tool.approval_requested'})


def client_event_filter():
    """A filter for one Discord socket (events.stream.stream_events). A turn is the bot's once an event of it carries
    source 'discord' (chat.input, model.started), before any of its deltas."""
    turns = set()
    def relevant(event):
        if event.payload.get('source') == 'discord' and event.turn_id:
            turns.add(event.turn_id)
            if len(turns) > 256: turns.clear(); turns.add(event.turn_id)
        return event.type in CLIENT_EVENTS or event.type in TURN_EVENTS and event.turn_id in turns
    return relevant


class DiscordChat(BaseModel):
    text: str = Field(min_length=1, max_length=16000)
    user_name: str = Field(default='Discord user', max_length=200)
    turn_id: UUID
    client_id: UUID | None = None
    user_id: str | None = Field(default=None, pattern=r'^[0-9]{1,20}$')
    channel_id: str | None = Field(default=None, pattern=r'^[0-9]{1,20}$')
    guild_id: str | None = Field(default=None, pattern=r'^[0-9]{1,20}$')
    message_id: str | None = Field(default=None, max_length=100)


class SpeechExport(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class StopTurn(BaseModel):
    turn_id: UUID


def create_router(get_session, get_launcher=None):
    router = APIRouter(prefix='/api/discord', tags=['discord'])

    def current():
        session = get_session()
        if session is None or not session.is_open: raise HTTPException(503, 'Companion runtime unavailable')
        return session

    @router.post('/chat')
    async def chat(request: DiscordChat):
        session = current()
        origin = {}
        if get_launcher:
            service = get_launcher()
            if str(request.client_id) != service.client_id: raise HTTPException(409, 'Connect the Discord client to this backend before sending messages')
            if not request.user_id or not request.channel_id or not service.access.settings().allows(int(request.user_id), int(request.channel_id), guild=request.guild_id is not None):
                raise HTTPException(403, 'Discord user/channel is not whitelisted')
            origin = {'source': 'discord', 'conversation_id': f'discord:{service.client_id}:{request.guild_id or "dm"}:{request.channel_id}',
                'user_id': request.user_id, 'channel_id': request.channel_id, 'guild_id': request.guild_id,
                'message_id': request.message_id or str(request.turn_id)}
        try:
            response = await run_in_threadpool(session.respond, request.text, request.user_name,
                speak=False, turn_id=str(request.turn_id), **({'origin': origin} if origin else {}))
            return {'text': response.message.content, 'turn_id': str(request.turn_id)}
        except TurnCancelled: return {'cancelled': True, 'turn_id': str(request.turn_id)}
        except TurnBusy as exc: raise HTTPException(409, 'Companion is busy; retry when its active turn finishes') from exc

    @router.post('/stop')
    def stop(request: StopTurn):
        return {'stopped': current().cancel_turn(str(request.turn_id))}  # never a turn other than this one

    @router.post('/transcribe')
    async def transcribe(request: Request):
        session = current()
        if request.headers.get('content-type', '').split(';')[0] != 'application/octet-stream':
            raise HTTPException(415, 'Send mono 16 kHz signed little-endian PCM16')
        pcm = bytearray()
        async for chunk in request.stream():
            if len(pcm) + len(chunk) > MAX_PCM_BYTES: raise HTTPException(413, 'Audio exceeds 60 seconds')
            pcm.extend(chunk)
        if not pcm or len(pcm) % 2: raise HTTPException(400, 'Empty or invalid PCM16')
        try: text = await run_in_threadpool(transcribe_pcm, session, bytes(pcm))
        except Exception as exc: raise HTTPException(503, 'Speech recognition unavailable; text chat remains usable') from exc
        return {'text': text}

    @router.post('/speech')
    async def speech(request: SpeechExport):
        try: audio = await run_in_threadpool(synthesize_wav, current().config, request.text)
        except Exception as exc: raise HTTPException(503, 'GPT-SoVITS audio export unavailable; try again on the next reply') from exc
        return Response(audio, media_type='audio/wav')

    return router
