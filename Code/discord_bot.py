"""Start the optional Discord client of the already-running Python backend."""
import asyncio
import logging
import os
from pathlib import Path
import sys
import threading


def exit_with_backend(bot):
    """Started by the desktop backend: stop when its pipe closes, which happens however the backend exits."""
    try: sys.stdin.read()
    except (OSError, ValueError): pass
    try: asyncio.run_coroutine_threadsafe(bot.close(), bot.loop).result(10)
    except Exception: pass  # not logged in yet, or already closing
    os._exit(0)


def main():
    from dotenv import load_dotenv
    from process.app_core.integrations.discord.access import DiscordAccess
    from process.app_core.integrations.discord.bot import CompanionBot
    root = Path(os.environ.get('RIKO_DATA_DIR', Path(__file__).resolve().parents[1]))
    os.environ.setdefault('RIKO_DATA_DIR', str(root))  # Where the backend client finds the API token.
    load_dotenv(root / '.env') # Never search parent/private directories for credentials.
    settings = DiscordAccess(root).settings()
    if not settings.token: raise ValueError('Discord_bot_token is required')
    if not settings.admins: raise ValueError('Configure Discord_admins before starting the bot')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    bot = CompanionBot(settings)
    if os.environ.get('RIKO_EXIT_WITH_BACKEND') == '1':
        threading.Thread(target=exit_with_backend, args=(bot,), name='backend-watch', daemon=True).start()
    bot.run(settings.token)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        # The launcher consumes only these fixed codes, never raw logs or tokens.
        name = type(exc).__name__
        code = 'login' if name == 'LoginFailure' else 'intents' if name == 'PrivilegedIntentsRequired' else 'dependency' if isinstance(exc, ImportError) else 'backend' if isinstance(exc, TimeoutError) else 'startup'
        print('RIKO_DISCORD_ERROR:' + code, flush=True)
        raise SystemExit(1) from None
