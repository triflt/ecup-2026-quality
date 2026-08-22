import re

from src.model import compose_text


_SPACE = re.compile(r"\s+")
_BAD = re.compile(r"(?:\bбад\b|биологически\s+активн|dietary\s+supplement|food\s+supplement)", re.I)
_SPORT = re.compile(r"(?:спортивн\w*\s+питан|\bbcaa\b|\bбцаа\b|l[- ]?карнитин|l[- ]?carnitine|\bпротеин\w*\b|аминокислот)", re.I)
_FIRE = re.compile(r"(?:спич|зажигал|огниво|легковоспламен|горюч|баллон\w*\s+с\s+газ|бензин|керосин|бутан|пропан)", re.I)
_DEVICE = re.compile(r"(?:мангал|гриль|газов\w*\s+плит|горелк|без\s+(?:газа|топлива|жидкости|угля|спичек))", re.I)


def comment_for(category, prediction, name, description):
    text = compose_text(name, description)
    if category == "БАД":
        if prediction and _BAD.search(text):
            comment = "В тексте присутствует прямая маркировка БАД или dietary supplement, поэтому заявленная категория товара подтверждается."
        elif prediction:
            comment = "Название и описание соответствуют биологически активной добавке, поэтому оснований считать категорию карточки ошибочной нет."
        elif _SPORT.search(text):
            comment = "Товар относится к спортивному питанию, которое по правилам не является БАД, поэтому категория карточки указана неверно."
        else:
            comment = "Прямая маркировка БАД или dietary supplement в тексте не подтверждена, поэтому заявленная категория карточки некорректна."
    else:
        if prediction and _FIRE.search(text):
            comment = "Товар содержит источник воспламенения либо горючее содержимое, поэтому заявленная категория карточки подтверждается."
        elif prediction:
            comment = "Описание товара соответствует правилам отнесения к легковоспламеняющимся, поэтому категория карточки указана корректно."
        elif _DEVICE.search(text):
            comment = "Описано устройство без самостоятельного горючего содержимого, поэтому категория легковоспламеняющихся указана неверно."
        else:
            comment = "Самостоятельный источник огня или горючее содержимое не подтверждены, поэтому заявленная категория карточки некорректна."
    comment = _SPACE.sub(" ", comment).strip().replace("<", " ").replace(">", " ")
    if len(comment) < 50:
        comment += " Решение принято по названию и описанию товара."
    return comment[:300].rsplit(" ", 1)[0] if len(comment) > 300 else comment


def format_result(category, prediction, name, description):
    verdict = "не бан" if int(prediction) == 1 else "бан"
    return f"<комментарий>{comment_for(category, prediction, name, description)}<вердикт>{verdict}"
