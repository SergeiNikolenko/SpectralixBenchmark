# G-Eval на ветке `benchmark_v3`

## Назначение

Этот документ описывает, как устроен `g_eval` на ветке `benchmark_v3`, зачем он нужен, где находится в коде и как встроен в evaluation pipeline.

## Краткое описание

На `benchmark_v3` `g_eval` используется как режим judge-а по умолчанию для открытых ответов.

Он нужен потому, что открытые химические ответы нельзя надёжно оценивать простым сравнением строк. Вместо свободного запроса "поставь score" pipeline ограничивает LLM-судью:

- заранее заданными критериями оценки,
- явными шагами проверки,
- фиксированной шкалой от `0` до `10`,
- строгой структурой выходного ответа.

Для типов ответов, где возможна жёсткая проверка, `g_eval` не используется. Там оценка считается обычной детерминированной логикой в коде.

## Зачем мы его используем

В pipeline есть два разных класса задач.

1. Задачи с точным ответом  
   Примеры: `single_choice`, `multiple_choice`, `ordering`, `numeric`, `msms_structure_prediction`.

   Для них надёжнее, дешевле и проще использовать deterministic scoring, а не LLM.

2. Задачи с открытым ответом  
   Примеры: `text`, `reaction_description`, `property_determination`, `full_synthesis`.

   Для них нужна смысловая оценка. Один и тот же правильный ответ может быть сформулирован по-разному, а частично правильные ответы должны получать частичный балл. Именно для этого и нужен `g_eval`.

Главная цель `g_eval` состоит в том, чтобы сделать LLM-based judging более управляемым, более понятным и менее "интуитивным", чем простой prompt формата "дай score от 0 до 1".

## Общий поток выполнения

На `benchmark_v3` оценка проходит так:

1. Student stage записывает `student_output.jsonl`.
2. `run_full_matrix.py` запускает judge stage.
3. `llm_judge.py` читает каждую строку и смотрит на `answer_type`.
4. Для детерминированных типов score считается кодом.
5. Для открытых типов вызывается `g_eval`.
6. `g_eval` возвращает rubric score от `0` до `10`.
7. Этот балл нормализуется в диапазон `[0.0, 1.0]`.
8. Нормализованный score умножается на `max_score`, и получается `final_score`.

## Где это находится в коде

### 1. Выбор judge-режима

`g_eval` задан как judge mode по умолчанию в matrix runner:

- `scripts/evaluation/run_full_matrix.py`

Здесь:

- `--judge-method` по умолчанию равен `g_eval`,
- выбранный режим передаётся в `run_llm_judge(...)`.

### 2. Основная логика judge-а

Главная маршрутизация находится в:

- `scripts/evaluation/llm_judge.py`

Этот файл определяет:

- какие типы ответов считаются детерминированно,
- какие типы идут в LLM judge,
- когда вызывать `g_eval`,
- когда делать fallback на standard structured judge,
- как считать итоговый балл строки.

### 3. Рубрики для G-Eval

Шаблоны rubric/specification лежат в:

- `scripts/evaluation/judge_rubrics.py`

Там заданы:

- общие критерии,
- критерии для отдельных answer types,
- шаги оценки,
- фиксированная шкала `0/2/4/6/8/10`.

### 4. Реальный вызов G-Eval

Сам вызов LLM через `g_eval` находится в:

- `scripts/pydantic_guard/judge_geval.py`

Этот модуль:

- создаёт structured judge agent,
- отправляет rubric-guided prompt,
- валидирует структурированный ответ,
- переводит rubric score `0..10` в score `0.0..1.0`.

### 5. Схема структурированного ответа

Контракт ответа задан в:

- `scripts/pydantic_guard/schemas.py`

Нужная схема называется `GEvalJudgeResult`.

## Как работает маршрутизация

Judge сначала нормализует `answer_type`, а затем делит строки на две группы.

### Детерминированная группа

Без LLM считаются:

- `single_choice`
- `multiple_choice`
- `ordering`
- `numeric`
- `msms_structure_prediction`

Используемые правила включают:

- strict token match,
- F1-подобную оценку для multiple choice,
- позиционное совпадение для ordering,
- проверку чисел по точному значению или диапазону,
- strict string match для SMILES-подобных ответов.

### Группа G-Eval

Все остальные open-ended answer types отправляются в `g_eval`.

Это основной путь для:

- `text`
- `reaction_description`
- `property_determination`
- `full_synthesis`

## Какой prompt строит G-Eval

`g_eval` не просит модель просто вернуть число без контекста.

Он строит prompt, в который входят:

- текст вопроса,
- canonical answer,
- student answer,
- критерии оценки,
- шаги оценки,
- rubric scale,
- инструкция вернуть строго структурированный ответ.

Смысл этого подхода в том, чтобы заставить judge-а оценивать ответ внутри заранее заданной рамки, а не выдавать score на основе размытого общего впечатления.

## Что должна вернуть модель

Модель обязана вернуть структурированный объект со следующими полями:

- `criteria_steps`
- `step_findings`
- `rubric_score_0_to_10`
- `llm_comment`

Это важно по двум причинам:

1. Невалидные ответы проще обнаруживать и ретраить.
2. Появляется audit trail, объясняющий, почему был поставлен конкретный балл.

## Как считается итоговый score

Сам `g_eval` возвращает rubric score от `0` до `10`.

Дальше pipeline считает:

- `llm_score = rubric_score_0_to_10 / 10.0`
- `final_score = max_score * llm_score`

Пример:

- у вопроса `max_score = 4`
- `g_eval` вернул `rubric_score_0_to_10 = 6`
- нормализованный `llm_score = 0.6`
- итоговый `final_score = 2.4`

## Fallback-поведение

Если `g_eval` не смог отработать, pipeline может переключиться на обычный structured judge.

Это поведение управляется флагом:

- `--judge-g-eval-fallback-structured`

На `benchmark_v3` fallback по умолчанию включён.

То есть judge stage устроен так, чтобы сначала предпочитать rubric-guided judging, но не зависеть полностью от успешной работы `g_eval` на каждой строке.

## Трейсы и отладка

Если включён trace logging, judge дописывает данные в per-question traces.

Для `g_eval` в трейсах можно увидеть:

- judge mode,
- score method,
- нормализованный score,
- final score,
- judge comment,
- criteria steps,
- step findings,
- rubric score от `0` до `10`.

Это одно из главных операционных преимуществ текущей реализации: решение judge-а можно разбирать заметно легче, чем в случае с plain free-form LLM judge.

## Важная деталь реализации

Хотя режим называется rubric-based, сама rubric не загружается из benchmark dataset.

На `benchmark_v3` строки benchmark содержат:

- `question_text`
- `canonical_answer`
- `max_score`
- служебные metadata fields

Но rubric для оценки берётся из кода в `scripts/evaluation/judge_rubrics.py` и выбирается по `answer_type`.

Иными словами:

- benchmark data задаёт правильный ответ и вес вопроса,
- код задаёт правила оценки.

То есть архитектурно это code-driven implementation, а не dataset-driven implementation.

## Сильные стороны

- Использует deterministic scoring там, где возможны жёсткие правила.
- Использует rubric-guided LLM judging там, где нужна смысловая оценка.
- Возвращает структурированный output вместо хрупкого свободного текста.
- Поддерживает fallback на standard structured judge.
- Даёт trace-level информацию для разбора judge decisions.

## Ограничения

- Rubric захардкожена по `answer_type`, а не хранится в benchmark rows.
- Качество judge-а всё равно зависит от качества модели.
- Type-level rubric может быть слишком грубой для отдельных семейств вопросов.
- Подход улучшает консистентность, но не делает оценивание полностью объективным.

## Итог

На `benchmark_v3` `g_eval` используется как основной semantic grading mode для открытых ответов.

Он нужен, чтобы сделать LLM judging более управляемым, более отлаживаемым и более консистентным. Реализация сочетает:

- deterministic scoring для точных типов задач,
- rubric-guided structured LLM judging для open-ended задач,
- fallback на обычный structured judge при сбое.

С архитектурной точки зрения `g_eval` не является отдельной подсистемой. Это специализированный judge mode внутри существующего evaluation pipeline.
