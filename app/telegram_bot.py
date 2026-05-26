"""Telegram bot interface for adoption speed prediction."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from petfinder.constants import DEFAULT_MODEL_PATH, WIZARD_FIELDS
from petfinder.defaults import merge_with_defaults
from petfinder.inference import Predictor
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

(
    TYPE,
    AGE,
    BREED1,
    GENDER,
    COLOR1,
    STERILIZED,
    HEALTH,
    VACCINATED,
    DEWORMED,
    FEE,
    PHOTOAMT,
    NAME,
) = range(12)

WIZARD_PROMPTS = {
    TYPE: (
        "Кто это?\n"
        "1 — собака\n"
        "2 — кошка\n"
        "3 — другое животное\n\n"
        "Отправьте цифру:"
    ),
    AGE: (
        "Сколько месяцев животному?\n"
    ),
    BREED1: (
        "Введите код породы животного из базы PetFinder.\n"
        "Если точный код не знаете, введите 266.\n\n"
        "Отправьте число:"
    ),
    GENDER: (
        "Пол:\n"
        "1 — мальчик\n"
        "2 — девочка\n"
        "3 — несколько животных\n\n"
        "Отправьте цифру:"
    ),
    COLOR1: (
        "Основной окрас:\n"
        "1 — чёрный\n"
        "2 — коричневый\n"
        "3 — золотистый\n"
        "4 — жёлтый\n"
        "5 — кремовый / светлый\n"
        "6 — серый\n"
        "7 — белый\n\n"
        "Отправьте цифру от 1 до 7:"
    ),
    STERILIZED: (
        "Стерилизовано / кастрировано?\n"
        "1 — да\n"
        "2 — нет\n"
        "3 — не знаю\n\n"
        "Отправьте цифру:"
    ),
    HEALTH: (
        "Состояние здоровья:\n"
        "1 — здоров(а), без проблем\n"
        "2 — есть небольшие проблемы (царапина, лёгкая болезнь)\n"
        "3 — серьёзные проблемы со здоровьем\n\n"
        "Отправьте цифру:"
    ),
    VACCINATED: (
        "Привит(а)?\n"
        "1 — да\n"
        "2 — нет\n"
        "3 — не знаю\n\n"
        "Отправьте цифру:"
    ),
    DEWORMED: (
        "Обработан(а) от глистов?\n"
        "1 — да\n"
        "2 — нет\n"
        "3 — не знаю\n\n"
        "Отправьте цифру:"
    ),
    FEE: (
        "Сколько стоит усыновление (в рингgit, валюта датасета)?\n"
        "0 — бесплатно, 100 — платное усыновление и т.д.\n\n"
        "Отправьте число:"
    ),
    PHOTOAMT: (
        "Сколько фотографий в объявлении?\n"
        "Например: 1, 3, 5"
    ),
    NAME: (
        "Как зовут животное?\n"
        "Напишите имя текстом. Если имени нет — отправьте символ -"
    ),
}

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(model_path=MODEL_PATH)


def _format_result(result) -> str:
    lines = [
        f"Предсказанный AdoptionSpeed: *{result.adoption_speed}*",
        f"Интерпретация: {result.class_label_ru}",
    ]
    if result.probabilities:
        top = sorted(
            result.probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        proba_str = ", ".join(f"класс {k}: {v:.1%}" for k, v in top)
        lines.append(f"Топ вероятностей: {proba_str}")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я оцениваю, как быстро животное из приюта может найти дом.\n\n"
        "Команды:\n"
        "/predict — ответить на простые вопросы об объявлении\n"
        "/predict_json — для продвинутых: вставить готовый JSON\n"
        "/cancel — прервать опрос\n\n"
        "В конце вы получите класс скорости пристройства (0 — очень быстро, "
        "4 — долго не пристраивается) и краткое пояснение."
    )


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["partial"] = {}
    await update.message.reply_text(
        "Начнём опрос про животное. На каждый вопрос отвечайте одним сообщением "
        "(цифрой или текстом, как указано ниже).\n"
        "Остальные детали объявления подставятся автоматически."
    )
    await update.message.reply_text(WIZARD_PROMPTS[TYPE])
    return TYPE


def _parse_numeric(value: str, field: str):
    try:
        if field == "PhotoAmt":
            return float(value)
        return int(value)
    except ValueError as exc:
        raise ValueError("Нужно отправить число. Попробуйте ещё раз.") from exc


async def _wizard_step(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    field_name: str,
    current_state: int,
    next_state: int,
) -> int:
    text = (update.message.text or "").strip()
    partial = context.user_data.setdefault("partial", {})

    if field_name == "Name":
        partial[field_name] = "" if text == "-" else text
    else:
        try:
            partial[field_name] = _parse_numeric(text, field_name)
        except ValueError as exc:
            await update.message.reply_text(str(exc))
            return current_state

    if next_state == ConversationHandler.END:
        return await _finish_wizard(update, context)

    await update.message.reply_text(WIZARD_PROMPTS[next_state])
    return next_state


async def _finish_wizard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    partial = context.user_data.get("partial", {})
    try:
        record = merge_with_defaults(partial)
        result = get_predictor().predict_one(record)
    except Exception as exc:
        await update.message.reply_text(f"Ошибка предсказания: {exc}")
        return ConversationHandler.END

    await update.message.reply_text(
        _format_result(result),
        parse_mode="Markdown",
    )
    context.user_data.pop("partial", None)
    return ConversationHandler.END


async def step_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Type", TYPE, AGE)


async def step_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Age", AGE, BREED1)


async def step_breed1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Breed1", BREED1, GENDER)


async def step_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Gender", GENDER, COLOR1)


async def step_color1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Color1", COLOR1, STERILIZED)


async def step_sterilized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Sterilized", STERILIZED, HEALTH)


async def step_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Health", HEALTH, VACCINATED)


async def step_vaccinated(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Vaccinated", VACCINATED, DEWORMED)


async def step_dewormed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Dewormed", DEWORMED, FEE)


async def step_fee(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "Fee", FEE, PHOTOAMT)


async def step_photoamt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(update, context, "PhotoAmt", PHOTOAMT, NAME)


async def step_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _wizard_step(
        update, context, "Name", NAME, ConversationHandler.END
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop("partial", None)
    await update.message.reply_text("Диалог отменён.")
    return ConversationHandler.END


async def predict_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправьте JSON с признаками животного (без AdoptionSpeed), "
        "как в POST /predict API."
    )
    context.user_data["awaiting_json"] = True


async def handle_json_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_json"):
        return
    context.user_data["awaiting_json"] = False
    try:
        record = json.loads(update.message.text)
        result = get_predictor().predict_one(record)
        await update.message.reply_text(
            _format_result(result),
            parse_mode="Markdown",
        )
    except json.JSONDecodeError:
        await update.message.reply_text("Некорректный JSON. Попробуйте снова /predict_json")
    except Exception as exc:
        await update.message.reply_text(f"Ошибка: {exc}")


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in environment or .env")

    application = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_start)],
        states={
            TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_type)],
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_age)],
            BREED1: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_breed1)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_gender)],
            COLOR1: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_color1)],
            STERILIZED: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, step_sterilized)
            ],
            HEALTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_health)],
            VACCINATED: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, step_vaccinated)
            ],
            DEWORMED: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_dewormed)],
            FEE: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_fee)],
            PHOTOAMT: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_photoamt)],
            NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, step_name)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(conv)
    application.add_handler(CommandHandler("predict_json", predict_json))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_json_message)
    )

    logger.info("Bot started. Wizard fields: %s", WIZARD_FIELDS)
    application.run_polling()


if __name__ == "__main__":
    main()
