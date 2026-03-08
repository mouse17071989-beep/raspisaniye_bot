import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

WEEKDAYS_RU = [
    "ПОНЕДЕЛЬНИК",
    "ВТОРНИК",
    "СРЕДА",
    "ЧЕТВЕРГ",
    "ПЯТНИЦА",
    "СУББОТА",
    "ВОСКРЕСЕНЬЕ",
]

SLOT_EMOJIS = ["🔷", "🟩", "🟥", "🟪", "🟨", "🟦", "🟧", "⬜️"]

GMOD_GAMES = [
    "ОДРП",
    "Залупинск",
    "CFC",
    "ОДРП — Admins",
    "Пензенск",
    "TFS",
]

OTHER_GAMES = [
    "VR",
    "City Car Driving",
    "TF2",
    "CS2",
    "Omsi 2",
    "CS:GO LEG (С КИБЕРСПОРТСМПЕН228)",
]

ALL_GAMES = GMOD_GAMES + OTHER_GAMES

DOUBLE_SLOT_GAMES = ["ОДРП", "Пензенск", "CFC", "TFS", "Atroll"]

CONFIG_PATH = Path("config.json")
AI_CONFIG_PATH = Path("ai_config.json")

# ---------------------------------------------------------------------------
# Конфиги
# ---------------------------------------------------------------------------


@dataclass
class Channel:
    name: str
    id: int


@dataclass
class AIConfig:
    groq_api_key: str
    groq_model: str
    temperature: float
    system_prompt: str


@dataclass
class BotConfig:
    telegram_token: str
    channels: List[Channel]
    active_channel: int
    autopost_time: str
    timezone: str

    @property
    def chat_id(self) -> int:
        return self.channels[self.active_channel].id

    @property
    def chat_name(self) -> str:
        return self.channels[self.active_channel].name


def load_ai_config() -> AIConfig:
    # Приоритет: переменные окружения → ai_config.json → значения по умолчанию
    groq_key = os.environ.get("GROQ_API_KEY", "")
    groq_model = os.environ.get("GROQ_MODEL", "")
    temperature = os.environ.get("GROQ_TEMPERATURE", "")
    system_prompt = os.environ.get("GROQ_SYSTEM_PROMPT", "")

    if AI_CONFIG_PATH.exists():
        with AI_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        groq_key = groq_key or data.get("groq_api_key", "")
        groq_model = groq_model or data.get("groq_model", "")
        temperature = temperature or str(data.get("temperature", ""))
        system_prompt = system_prompt or data.get("system_prompt", "")

    if not groq_key:
        raise ValueError("GROQ_API_KEY не задан (env или ai_config.json)")

    return AIConfig(
        groq_api_key=groq_key,
        groq_model=groq_model or "llama-3.3-70b-versatile",
        temperature=float(temperature) if temperature else 0.8,
        system_prompt=system_prompt
        or "Ты генератор расписания игровых слотов. Возвращай только JSON без markdown и без пояснений.",
    )


def load_config() -> "BotConfig":
    # Приоритет: переменные окружения → config.json → значения по умолчанию
    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("CHAT_ID", "")
    channel_name = os.environ.get("CHANNEL_NAME", "Канал")
    autopost_time = os.environ.get("AUTOPOST_TIME", "")
    timezone = os.environ.get("TIMEZONE", "")

    channels: List[Channel] = []

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        token = token or data.get("telegram_token", "")
        autopost_time = autopost_time or data.get("autopost_time", "")
        timezone = timezone or data.get("timezone", "")

        if not chat_id:
            channels_raw = data.get("channels", [])
            if channels_raw:
                channels = [Channel(name=ch["name"], id=int(ch["id"])) for ch in channels_raw]
            else:
                cid = int(data.get("chat_id", data.get("group_id", 0)))
                channels = [Channel(name="Канал", id=cid)]

    if not token:
        raise ValueError("TELEGRAM_TOKEN не задан (env или config.json)")

    if chat_id and not channels:
        channels = [Channel(name=channel_name, id=int(chat_id))]
    if not channels:
        raise ValueError("CHAT_ID не задан (env или config.json)")

    return BotConfig(
        telegram_token=token,
        channels=channels,
        active_channel=0,
        autopost_time=autopost_time or "22:00",
        timezone=timezone or "Asia/Novosibirsk",
    )


def save_config(cfg: BotConfig) -> None:
    data = {
        "telegram_token": cfg.telegram_token,
        "channels": [{"name": ch.name, "id": ch.id} for ch in cfg.channels],
        "active_channel": cfg.active_channel,
        "autopost_time": cfg.autopost_time,
        "timezone": cfg.timezone,
    }
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_ai_config(ai_cfg: AIConfig) -> None:
    data = {
        "groq_api_key": ai_cfg.groq_api_key,
        "groq_model": ai_cfg.groq_model,
        "temperature": ai_cfg.temperature,
        "system_prompt": ai_cfg.system_prompt,
    }
    with AI_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Генератор расписания (Groq)
# ---------------------------------------------------------------------------


class GroqScheduleGenerator:
    def __init__(self, ai_cfg: AIConfig) -> None:
        self.ai_cfg = ai_cfg
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def reload(self, ai_cfg: AIConfig) -> None:
        self.ai_cfg = ai_cfg

    @staticmethod
    def _extract_json_object(text: str) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("В ответе модели не найден JSON-объект")
        return match.group(0)

    @staticmethod
    def _extract_slots(response_text: str) -> List[str]:
        payload = json.loads(response_text)
        slots = payload.get("slots")
        if not isinstance(slots, list):
            raise ValueError("Поле 'slots' отсутствует или имеет неверный тип")
        return [str(slot).strip() for slot in slots]

    @staticmethod
    def _validate_slots(slots: List[str]) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        if len(slots) != 8:
            violations.append("Должно быть ровно 8 слотов")
            return False, violations

        allowed = set(ALL_GAMES + DOUBLE_SLOT_GAMES)
        for idx, game in enumerate(slots, start=1):
            if game not in allowed:
                violations.append(f"Слот {idx}: игра '{game}' не из разрешённого списка")
        if violations:
            return False, violations

        if slots[2] not in GMOD_GAMES:
            violations.append("Слот 3 должен быть игрой из GMod")

        if slots[4] not in DOUBLE_SLOT_GAMES:
            violations.append(
                f"Слот 5 должен быть одной из: {', '.join(DOUBLE_SLOT_GAMES)}"
            )
        if slots[5] not in DOUBLE_SLOT_GAMES:
            violations.append(
                f"Слот 6 должен быть одной из: {', '.join(DOUBLE_SLOT_GAMES)}"
            )
        if slots[4] != slots[5]:
            violations.append("Слоты 5 и 6 должны быть одинаковыми (сдвоенные)")

        if "VR" in slots[:4]:
            violations.append("VR не должен быть в слотах 1-4")

        if slots.count("ОДРП — Admins") > 1:
            violations.append("ОДРП — Admins может быть максимум 1 раз в дне")

        return len(violations) == 0, violations

    async def _ask_groq(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.ai_cfg.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.ai_cfg.groq_model,
            "messages": [
                {"role": "system", "content": self.ai_cfg.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.ai_cfg.temperature,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            for attempt in range(4):
                response = await client.post(self.api_url, headers=headers, json=payload)
                if response.status_code == 429:
                    wait = 3 * (attempt + 1)
                    logging.warning("Groq 429 rate limit, жду %d сек...", wait)
                    await asyncio.sleep(wait)
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"Groq error {response.status_code}: {response.text}")
                break
            else:
                raise RuntimeError("Groq: 429 после 4 попыток, рейтлимит не снялся")
            data = response.json()

        content = data["choices"][0]["message"].get("content")
        if isinstance(content, str):
            return content
        raise ValueError("Groq вернул неожиданный формат ответа")

    async def generate_day_slots(self, day_name: str) -> List[str]:
        double_list = json.dumps(DOUBLE_SLOT_GAMES, ensure_ascii=False)
        base_prompt = (
            f"Сгенерируй расписание на {day_name}.\n"
            "Разрешённые игры:\n"
            f"{json.dumps(ALL_GAMES + [g for g in DOUBLE_SLOT_GAMES if g not in ALL_GAMES], ensure_ascii=False)}\n\n"
            "Правила:\n"
            "1) Всего 8 слотов\n"
            "2) Слот 3 обязательно GMod\n"
            f"3) Слоты 5 и 6 — СДВОЕННЫЕ, одна и та же игра из списка: {double_list}\n"
            "4) VR только в поздних слотах (5-8)\n"
            "5) ОДРП — Admins максимум 1 раз\n"
            "6) CS:GO LEG (С КИБЕРСПОРТСМПЕН228) — РЕДКАЯ игра, НЕ ставь её каждый раз. Только изредка и только в слот 8\n"
            "7) Слот 8 — любая игра из списка, НЕ обязательно CS:GO\n"
            "8) Повторы допустимы, но старайся разнообразить\n\n"
            'Верни строго JSON вида: {"slots":["игра1",...,"игра8"]}'
        )

        last_violations: List[str] = []
        current_prompt = base_prompt

        for _ in range(3):
            response_text = await self._ask_groq(current_prompt)
            json_text = self._extract_json_object(response_text)
            slots = self._extract_slots(json_text)
            ok, violations = self._validate_slots(slots)
            if ok:
                return slots

            last_violations = violations
            current_prompt = (
                base_prompt
                + "\n\nТвой прошлый ответ нарушил правила:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n\nИсправь и верни только корректный JSON."
            )

        raise RuntimeError(
            "Groq не смог сгенерировать валидное расписание после 3 попыток: "
            + "; ".join(last_violations)
        )

    async def generate_greeting(self, day_name: str) -> str:
        prompt = (
            f"Напиши атмосферное вступление к игровому расписанию на {day_name}. "
            "Стиль: короткие кинематографичные фразы, каждая с новой строки. "
            "Как будто запускается игровой день. Пример стиля:\n"
            "Серверы прогреваются.\n"
            "Чаты оживают.\n"
            "Новая игровая неделя начинается.\n\n"
            f"Сегодня — {day_name.lower()}.\n"
            "Время открывать лобби, запускать карты и занимать свои слоты.\n\n"
            "Придумай свой уникальный вариант (не копируй пример дословно). "
            "3-6 коротких строк. Можно с эмодзи, но не перебарщивай. "
            "Упомяни день недели. Без markdown."
        )
        try:
            return await self._ask_groq(prompt)
        except Exception:
            return "🎮 Хорошего дня и удачных катк!"


# ---------------------------------------------------------------------------
# Бот
# ---------------------------------------------------------------------------


# Ключи настроек для user_data
SETTING_KEY = "awaiting_setting"

SETTINGS_MAP = {
    "set_model": ("🤖 Модель", "Введи название модели Groq (например llama-3.3-70b-versatile):"),
    "set_temp": ("🌡 Temperature", "Введи temperature (0.0 — 2.0):"),
    "set_prompt": ("📝 System prompt", "Введи новый system prompt:"),
    "set_time": ("⏰ Время автопоста", "Введи время в формате HH:MM (например 22:00):"),
    "set_tz": ("🌍 Таймзона", "Введи таймзону (например Asia/Novosibirsk):"),
    "set_add_ch": ("➕ Добавить канал", "Введи ID и название канала через пробел:\nНапример: -1001234567890 Мой канал"),
    "set_del_ch": ("🗑 Удалить канал", None),
}


def main_keyboard(config: BotConfig) -> InlineKeyboardMarkup:
    ch = config.channels[config.active_channel]
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📅 Расписание на завтра", callback_data="gen_tomorrow")],
            [InlineKeyboardButton("🗓 Выбрать день", callback_data="pick_day")],
            [InlineKeyboardButton("📨 Тест канала", callback_data="test_channel")],
            [
                InlineKeyboardButton(
                    f"📍 Канал: {ch.name}", callback_data="pick_channel"
                )
            ],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings_menu")],
        ]
    )


def settings_keyboard(config: BotConfig, ai_cfg: AIConfig) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"🤖 Модель: {ai_cfg.groq_model}", callback_data="set_model")],
            [InlineKeyboardButton(f"🌡 Temperature: {ai_cfg.temperature}", callback_data="set_temp")],
            [InlineKeyboardButton("📝 System prompt", callback_data="set_prompt")],
            [InlineKeyboardButton(f"⏰ Автопост: {config.autopost_time}", callback_data="set_time")],
            [InlineKeyboardButton(f"🌍 Таймзона: {config.timezone}", callback_data="set_tz")],
            [InlineKeyboardButton("➕ Добавить канал", callback_data="set_add_ch")],
            [InlineKeyboardButton("🗑 Удалить канал", callback_data="set_del_ch")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")],
        ]
    )


def delete_channel_keyboard(config: BotConfig) -> InlineKeyboardMarkup:
    rows = []
    for idx, ch in enumerate(config.channels):
        rows.append([InlineKeyboardButton(f"❌ {ch.name} ({ch.id})", callback_data=f"delch_{idx}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings_menu")])
    return InlineKeyboardMarkup(rows)


def day_picker_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for idx, name in enumerate(WEEKDAYS_RU):
        rows.append([InlineKeyboardButton(name, callback_data=f"day_{idx}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)


def channel_picker_keyboard(config: BotConfig) -> InlineKeyboardMarkup:
    rows = []
    for idx, ch in enumerate(config.channels):
        marker = "✅ " if idx == config.active_channel else ""
        rows.append(
            [InlineKeyboardButton(f"{marker}{ch.name}", callback_data=f"ch_{idx}")]
        )
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)


class ScheduleBot:
    def __init__(self, config: BotConfig, ai_cfg: AIConfig, tz: ZoneInfo) -> None:
        self.config = config
        self.ai_cfg = ai_cfg
        self.tz = tz
        self.generator = GroqScheduleGenerator(ai_cfg)

    async def build_day_message(self, weekday_index: int) -> str:
        day_name = WEEKDAYS_RU[weekday_index]
        slots = await self.generator.generate_day_slots(day_name)
        greeting = await self.generator.generate_greeting(day_name)
        lines = [greeting, "", f"🌈 {day_name}", ""]
        for index, game in enumerate(slots, start=1):
            lines.append(f"{SLOT_EMOJIS[index - 1]} СЛОТ {index}  →  {game}")
            lines.append("")
        return "\n".join(lines).strip()

    def _tomorrow_weekday(self) -> int:
        now = datetime.now(self.tz)
        tomorrow = now + timedelta(days=1)
        return tomorrow.weekday()

    # ---- Команды ----

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "🎮 Сервера онлайн. Бот готов к работе.\nВыбирай действие и поехали 🚀",
            reply_markup=main_keyboard(self.config),
        )

    async def generate_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        weekday = self._tomorrow_weekday()
        try:
            text = await self.build_day_message(weekday)
        except Exception as e:
            logging.exception("Ошибка генерации")
            await update.message.reply_text(f"❌ Ошибка генерации: {e}")
            return
        await update.message.reply_text(text, reply_markup=main_keyboard(self.config))

    # ---- Кнопки ----

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        data = query.data

        # --- Генерация на завтра ---
        if data == "gen_tomorrow":
            weekday = self._tomorrow_weekday()
            await query.message.reply_text("🔧 Собираю расписание... Секунду.")
            try:
                text = await self.build_day_message(weekday)
            except Exception as e:
                logging.exception("Ошибка генерации")
                await query.message.reply_text(f"❌ Сервер не ответил: {e}")
                return
            # Отправляем в канал
            try:
                await context.bot.send_message(chat_id=self.config.chat_id, text=text)
                await query.message.reply_text(
                    f"📡 Расписание улетело в «{self.config.chat_name}» ✅",
                    reply_markup=main_keyboard(self.config),
                )
            except Exception as e:
                logging.exception("Ошибка отправки в канал")
                await query.message.reply_text(
                    f"❌ Канал недоступен: {e}\n\nРасписание:\n{text}",
                    reply_markup=main_keyboard(self.config),
                )

        # --- Выбор дня ---
        elif data == "pick_day":
            await query.message.reply_text(
                "📅 На какой день собрать расписание?", reply_markup=day_picker_keyboard()
            )

        elif data.startswith("day_"):
            weekday = int(data.split("_")[1])
            await query.message.reply_text("🔧 Собираю расписание... Секунду.")
            try:
                text = await self.build_day_message(weekday)
            except Exception as e:
                logging.exception("Ошибка генерации")
                await query.message.reply_text(f"❌ Не получилось: {e}")
                return
            try:
                await context.bot.send_message(chat_id=self.config.chat_id, text=text)
                await query.message.reply_text(
                    f"📡 Расписание на {WEEKDAYS_RU[weekday]} улетело в «{self.config.chat_name}» ✅",
                    reply_markup=main_keyboard(self.config),
                )
            except Exception as e:
                logging.exception("Ошибка отправки в канал")
                await query.message.reply_text(
                    f"❌ Канал недоступен: {e}\n\nРасписание:\n{text}",
                    reply_markup=main_keyboard(self.config),
                )

        # --- Тест канала ---
        elif data == "test_channel":
            try:
                await context.bot.send_message(
                    chat_id=self.config.chat_id,
                    text="✅ Тестовое сообщение от бота расписания!",
                )
                await query.message.reply_text(
                    f"✅ Связь с «{self.config.chat_name}» есть!",
                    reply_markup=main_keyboard(self.config),
                )
            except Exception as e:
                logging.exception("Ошибка отправки в канал")
                await query.message.reply_text(
                    f"❌ Не удалось: {e}",
                    reply_markup=main_keyboard(self.config),
                )

        # --- Выбор канала ---
        elif data == "pick_channel":
            await query.message.reply_text(
                "📡 Куда публиковать расписание?",
                reply_markup=channel_picker_keyboard(self.config),
            )

        elif data.startswith("ch_"):
            idx = int(data.split("_")[1])
            if 0 <= idx < len(self.config.channels):
                self.config.active_channel = idx
                save_config(self.config)
                await query.message.reply_text(
                    f"✅ Переключился на «{self.config.chat_name}»",
                    reply_markup=main_keyboard(self.config),
                )

        # --- Меню настроек ---
        elif data == "settings_menu":
            context.user_data.pop(SETTING_KEY, None)
            await query.message.reply_text(
                "⚙️ Настройки бота",
                reply_markup=settings_keyboard(self.config, self.ai_cfg),
            )

        # --- Кнопки настроек (ожидание ввода) ---
        elif data in SETTINGS_MAP:
            label, prompt_text = SETTINGS_MAP[data]
            if data == "set_del_ch":
                if len(self.config.channels) <= 1:
                    await query.message.reply_text(
                        "❌ Нельзя удалить единственный канал.",
                        reply_markup=settings_keyboard(self.config, self.ai_cfg),
                    )
                else:
                    await query.message.reply_text(
                        "Какой канал удалить?",
                        reply_markup=delete_channel_keyboard(self.config),
                    )
            else:
                context.user_data[SETTING_KEY] = data
                await query.message.reply_text(
                    prompt_text,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("❌ Отмена", callback_data="settings_menu")]]
                    ),
                )

        # --- Удаление канала ---
        elif data.startswith("delch_"):
            idx = int(data.split("_")[1])
            if 0 <= idx < len(self.config.channels) and len(self.config.channels) > 1:
                removed = self.config.channels.pop(idx)
                if self.config.active_channel >= len(self.config.channels):
                    self.config.active_channel = 0
                save_config(self.config)
                await query.message.reply_text(
                    f"✅ Канал «{removed.name}» удалён.",
                    reply_markup=settings_keyboard(self.config, self.ai_cfg),
                )
            else:
                await query.message.reply_text(
                    "❌ Не удалось удалить.",
                    reply_markup=settings_keyboard(self.config, self.ai_cfg),
                )

        # --- Назад ---
        elif data == "back_main":
            context.user_data.pop(SETTING_KEY, None)
            await query.message.reply_text(
                "🎮 Готов к работе. Выбирай:", reply_markup=main_keyboard(self.config)
            )

    # ---- Обработка текстового ввода настроек ----

    async def text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        setting = context.user_data.pop(SETTING_KEY, None)
        if not setting:
            return

        value = update.message.text.strip()
        if not value:
            await update.message.reply_text("❌ Пустое значение, попробуй ещё раз.")
            context.user_data[SETTING_KEY] = setting
            return

        try:
            if setting == "set_model":
                self.ai_cfg.groq_model = value
                save_ai_config(self.ai_cfg)
                self.generator.reload(self.ai_cfg)
                reply = f"✅ Модель: {value}"

            elif setting == "set_temp":
                temp = float(value)
                if not 0.0 <= temp <= 2.0:
                    raise ValueError("Temperature должна быть от 0.0 до 2.0")
                self.ai_cfg.temperature = temp
                save_ai_config(self.ai_cfg)
                self.generator.reload(self.ai_cfg)
                reply = f"✅ Temperature: {temp}"

            elif setting == "set_prompt":
                self.ai_cfg.system_prompt = value
                save_ai_config(self.ai_cfg)
                self.generator.reload(self.ai_cfg)
                reply = f"✅ System prompt обновлён"

            elif setting == "set_time":
                parts = value.split(":")
                if len(parts) != 2:
                    raise ValueError("Формат: HH:MM")
                h, m = int(parts[0]), int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError("Некорректное время")
                self.config.autopost_time = value
                save_config(self.config)
                # Перезапускаем daily job
                jobs = context.job_queue.get_jobs_by_name("daily_schedule")
                for job in jobs:
                    job.schedule_removal()
                new_time = parse_post_time(value, self.tz)
                context.job_queue.run_daily(
                    self.scheduled_post, time=new_time, name="daily_schedule"
                )
                reply = f"✅ Автопост: {value}"

            elif setting == "set_tz":
                try:
                    new_tz = ZoneInfo(value)
                except (ZoneInfoNotFoundError, KeyError):
                    raise ValueError(f"Таймзона '{value}' не найдена")
                self.tz = new_tz
                self.config.timezone = value
                save_config(self.config)
                reply = f"✅ Таймзона: {value}"

            elif setting == "set_add_ch":
                # Формат: -100123456 Название канала
                first_space = value.find(" ")
                if first_space == -1:
                    raise ValueError("Формат: ID Название\nНапример: -1001234567890 Мой канал")
                ch_id = int(value[:first_space])
                ch_name = value[first_space + 1:].strip()
                if not ch_name:
                    raise ValueError("Укажи название канала после ID")
                self.config.channels.append(Channel(name=ch_name, id=ch_id))
                save_config(self.config)
                reply = f"✅ Канал «{ch_name}» ({ch_id}) добавлен"

            else:
                reply = "❓ Неизвестная настройка"

        except ValueError as e:
            await update.message.reply_text(
                f"❌ {e}",
                reply_markup=settings_keyboard(self.config, self.ai_cfg),
            )
            return

        await update.message.reply_text(
            reply, reply_markup=settings_keyboard(self.config, self.ai_cfg)
        )

    # ---- Автопостинг (22:00 → расписание на завтра) ----

    async def scheduled_post(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        weekday = self._tomorrow_weekday()
        try:
            text = await self.build_day_message(weekday)
            await context.bot.send_message(chat_id=self.config.chat_id, text=text)
            logging.info("Автопост: расписание на %s отправлено", WEEKDAYS_RU[weekday])
        except Exception as e:
            logging.exception("Ошибка автопостинга: %s", e)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------


def parse_post_time(raw_time: str, tz: ZoneInfo) -> time:
    parts = raw_time.split(":")
    if len(parts) != 2:
        raise ValueError("autopost_time должен быть в формате HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("autopost_time содержит некорректное время")
    return time(hour=hour, minute=minute, tzinfo=tz)


def create_timezone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logging.warning("Таймзона '%s' не найдена, использую UTC", tz_name)
        return ZoneInfo("UTC")


def build_application(config: BotConfig, ai_cfg: AIConfig) -> Application:
    tz = create_timezone(config.timezone)
    schedule_bot = ScheduleBot(config, ai_cfg, tz)

    app = ApplicationBuilder().token(config.telegram_token).build()

    app.add_handler(CommandHandler("start", schedule_bot.start_command))
    app.add_handler(CommandHandler("generate", schedule_bot.generate_command))
    app.add_handler(CallbackQueryHandler(schedule_bot.button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, schedule_bot.text_handler))

    post_time = parse_post_time(config.autopost_time, tz)
    app.job_queue.run_daily(schedule_bot.scheduled_post, time=post_time, name="daily_schedule")

    return app


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    config = load_config()
    ai_cfg = load_ai_config()
    app = build_application(config, ai_cfg)

    logging.info("Бот запущен. Автопост в %s на завтрашний день.", config.autopost_time)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
