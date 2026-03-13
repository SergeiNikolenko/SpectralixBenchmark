# Конструкция бенчмарка

Сейчас бенчмарк устроен как **трёхуровневая лестница задач по глубине планирования**, а не как один смешанный chemistry dataset.

Логика такая:

- `Level A` проверяет `reaction understanding`
- `Level B` проверяет `single-step retrosynthesis`
- `Level C` проверяет `multi-step synthesis planning`

Это сделано специально под гипотезу про `planning depth`: чем глубже горизонт планирования, тем сложнее задача для модели.

## Как он сейчас выглядит

У нас есть **два слоя данных**.

## 1. Большие agent pools

Это большие рабочие наборы для массовой раздачи задач агентам:

- `benchmark/level_a.jsonl`
- `benchmark/level_b.jsonl`
- `benchmark/level_c.jsonl`

Их размеры сейчас такие:

- `Level A`: `2,240,288` записей
- `Level B`: `3,372,962` записей
- `Level C`: `20,013` записей

## 2. Компактные paper eval subsets

Это маленькие детерминированные поднаборы для статьи и контролируемых сравнений:

- `benchmark/level_a_eval.jsonl` — `420`
- `benchmark/level_b_eval.jsonl` — `420`
- `benchmark/level_c_eval.jsonl` — `150`

То есть большие файлы нужны для throughput и агентных прогонов, а маленькие — для paper evaluation.

## Почему мы вообще пересобрали benchmark

Изначально в репозитории был `benchmark/benchmark_v1_0.jsonl`, но это был внутренний пилот. Он смешивал:

- общую органику
- `structure prediction`
- `MS/MS`
- `reaction understanding`
- `synthesis tasks`

Для paper это плохо, потому что там нет чистой оси сложности. Поэтому мы ушли от “одного общего chemistry benchmark” к `benchmark ladder`:

- `A`: локальное понимание реакции
- `B`: один ретрошаг назад
- `C`: глобальное `route planning`

Это и есть backbone всего нового benchmark design.

## Как мы его собирали

Мы не искали один “идеальный датасет”. Мы сделали `source-based normalization pipeline`.

Сначала собрали внешние источники в `external_sources/`, потом для каждого источника решили:

- он идёт напрямую в benchmark
- или он только `raw_material`
- или он `blocked/commercial` и только документируется

После этого написали нормализаторы, которые приводят разные форматы к одной схеме JSONL.

Ключевая идея была такая:

- не тащить чужие форматы как есть
- а превращать каждый источник в единый `internal task schema`

## Какие источники мы использовали

## Level A: Reaction Understanding

В `Level A` вошли:

- `PMechDB`
- `USPTO-50K`
- `ChEMU 2020`
- `CHORISO`
- `WEAVE2`

### Почему именно они

#### PMechDB

- это самый прямой источник для `mechanistic/reaction-center` задач
- поэтому он стал backbone для `reaction_center_identification` и `mechanistic_classification`

#### USPTO-50K

- хороший источник для `reaction/transformation classification`
- он дал более чистый `classification signal`

#### ChEMU 2020

- это не готовый benchmark на `reaction understanding`
- но он полезен как источник `reagent-role extraction` из `procedure text`
- поэтому мы его не взяли “как есть”, а `reframed` в benchmark-задачи

#### CHORISO

- расширяет покрытие `public reaction data`
- его тоже использовали не как исходный `task benchmark`, а как материал под normalized `Level A` rows

#### WEAVE2

- `procedural patent annotations`
- полезен для `reagent/procedure role coverage`
- тоже включён через переразметку в единую схему

### Пример задачи из Level A

- `record_id`: `pmechdb_manual_test_challenging_00001`
- `input_text`: `Identify the reaction center for the mapped transformation.`
- что видит агент:
  - `reaction_smirks`
  - `reactants`
  - `reagents`
  - `products`
- что должен вернуть агент:
  - `reaction_center`
  - `mechanistic_class`
  - `transformation_type`
  - при необходимости `major_product`

По смыслу это задача вида: “посмотри на одно превращение и объясни, где именно в молекуле происходит химическое изменение и к какому механистическому типу оно относится”.

Более подробно этот кейс выглядит так:

- во `input.reaction_smirks` хранится atom-mapped реакция, например:
  - `[Li:11][CH2:10]CCC[CH:20]=[CH:21][C:22](=[O:23])OC(C)(C)C.CCCCI>>...`
- это позволяет не просто угадать класс реакции по шаблону, а буквально проследить:
  - какие атомы меняют связи
  - где образуется новая связь
  - где исчезает старая связь
  - как меняется электронная структура в области карбонила и алкена
- в `gold.reaction_center` целевой ответ хранится как список changed bonds / changed atoms:
  - `10,11=10,20`
  - `20,21=21,22`
  - `22,23=23`
- в этом конкретном примере `gold.major_product` уже сохранён явно, поэтому задачу можно оценивать не только по локальному центру реакции, но и по корректности итогового результата
- `gold.mechanistic_class` и `gold.transformation_type` здесь могут быть `null`, то есть не каждый источник даёт одинаково богатую разметку

То есть practically это кейс на локальное химическое понимание:

- модель должна прочитать mapped transformation
- локализовать реакционный центр
- понять, какие связи перестраиваются
- при наличии дополнительных полей соотнести это с mechanistic class или product outcome

Минимальная рабочая форма ответа для такого кейса выглядит так:

```json
{
  "reaction_center": ["10,11=10,20", "20,21=21,22", "22,23=23"],
  "mechanistic_class": null,
  "transformation_type": null,
  "major_product": "..."
}
```

### Итог по Level A

Хорошо покрывает:

- `transformation_type`
- `reaction_center`
- `reagent_roles`
- `mechanistic_class` частично

Хуже покрывает:

- `selectivity explanation`
- `stereochemical implication`
- полноценное `mechanistic narration`

## Level B: Single-Step Retrosynthesis

В `Level B` вошли:

- `ORDerly retrosynthesis`
- `PaRoutes selected_reactions_all`

### Почему

#### ORDerly

- это чистый `open source one-step retrosynthesis source`
- хороший `benchmark-like backbone` для `immediate precursor prediction`

#### PaRoutes selected_reactions_all

- это уже не “маленький аккуратный benchmark”, а большой `one-step reaction pool`
- он был нужен, чтобы сильно расширить покрытие для agent solving

Здесь важно:

- `ORDerly` даёт cleaner benchmark rows
- `PaRoutes selected_reactions_all` даёт масштаб и разнообразие

### Пример задачи из Level B

- `record_id`: `orderly_retro_test_0543343`
- `input_text`: `Propose plausible immediate precursors for the target molecule.`
- `target`: `COc1cnn(-c2cc(Cl)cc(Cl)c2)c(=O)c1`
- что видит агент:
  - только целевую молекулу
- что должен вернуть агент:
  - `precursor_set`
  - `proposed_disconnection`
  - `key_transformation`
  - `justification`
  - `constraints`

По смыслу это задача вида: “разложи target на правдоподобные непосредственные прекурсоры и объясни, какой one-step disconnection ты предлагаешь”.

Более подробно этот кейс выглядит так:

- во входе есть только:
  - `input.target = COc1cnn(-c2cc(Cl)cc(Cl)c2)c(=O)c1`
- это означает, что агент не получает готовые reactants, reagents или route hints
- он должен сам предложить immediate precursor hypothesis
- в `gold.precursor_set` для этого примера лежит один ключевой предшественник:
  - `COc1cnn(-c2cc(Cl)cc(Cl)c2)c(=O)c1Br`
- отдельно в `gold.constraints` сохранён процедурный контекст, который можно использовать как ограничение или дополнительную проверку:
  - `agents`: `[Cl-]`, `[Li]CCCC`, `[NH4+]`
  - `solvents`: `C1CCOC1`
- в `metadata.procedure_details` дополнительно лежит текст процедуры, из которого видно, что это реальная reaction record, а не синтетически придуманный precursor pair

Что это значит с точки зрения reasoning:

- модель должна посмотреть на target
- понять, какой bond disconnection здесь наиболее правдоподобен
- предложить ближайший precursor set
- учесть, что в gold могут быть не только сами прекурсоры, но и procedural constraints
- в richer settings эта же задача позволяет проверять не просто `precursor_set`, а ещё и осмысленность объяснения

Минимальная рабочая форма ответа для такого кейса выглядит так:

```json
{
  "precursor_set": [
    "COc1cnn(-c2cc(Cl)cc(Cl)c2)c(=O)c1Br"
  ],
  "proposed_disconnection": null,
  "key_transformation": null,
  "justification": null,
  "constraints": {
    "agents": ["[Cl-]", "[Li]CCCC", "[NH4+]"],
    "solvents": ["C1CCOC1"]
  }
}
```

То есть это уже не вопрос “что происходит в реакции”, а вопрос “какой один шаг назад наиболее разумен для данного target”.

### Итог по Level B

Хорошо покрывает:

- `precursor_set`
- `local disconnection hints`
- `structural constraints`

Слабее покрывает:

- `rich justification`
- `named transformation explanation`
- `selectivity-aware retrosynthetic reasoning`

## Level C: Multi-Step Synthesis Planning

В `Level C` вошли:

- `PaRoutes n1`
- `PaRoutes n5`
- строгий subset из `benchmark_v1_0.jsonl`

### Почему

#### PaRoutes n1/n5

- это `route trees`
- они идеально ложатся на `route planning benchmark`
- дают:
  - `route depth`
  - `branching`
  - `convergence`
  - `reference route structure`

#### pilot subset из benchmark_v1_0

- не весь пилот
- а только задачи, которые реально требуют `synthesis planning / route proposal`
- это было важно, чтобы сохранить “олимпиадно-учебный” высокосигнальный слой

### Что мы сознательно не включали в Level C

- `forward multi-step execution`
- `product-sequence recovery`
- прочие старые pilot-задачи, которые не являются именно `route design`

### Пример задачи из Level C

- `record_id`: `paroutes_n1_00177`
- `input_text`: `Propose a multi-step synthesis route for the target molecule.`
- `target`: `Cc1ccc(C(=O)NC2CC2)cc1-c1ccc2c(C#N)nncc2c1`
- что видит агент:
  - только целевую молекулу
- что должен вернуть агент:
  - `reference_route`-совместимый маршрут или сопоставимый route proposal
  - `route_depth`
  - набор `reaction_steps`
  - информацию о `branching`
  - перечень `terminal_molecules`

По смыслу это задача вида: “спланируй полноценный синтетический маршрут до target, а не просто предложи один локальный шаг”.

Более подробно этот кейс выглядит так:

- во входе есть только:
  - `input.target = Cc1ccc(C(=O)NC2CC2)cc1-c1ccc2c(C#N)nncc2c1`
- в `gold.reference_route` хранится не строка с ответом, а целое дерево маршрута
- это дерево содержит:
  - molecule nodes
  - reaction nodes
  - nested children для прекурсоров каждого шага
- для данного примера отдельно вынесены агрегированные route-level поля:
  - `route_depth`
  - `reaction_steps`
  - `branching_reaction_nodes`
  - `terminal_molecules`
  - `terminal_in_stock`

То есть benchmark может оценивать сразу несколько уровней качества:

- нашла ли модель корректную общую route idea
- совпадает ли глубина маршрута с reference route
- умеет ли модель работать с convergent / branched synthesis
- выходят ли terminal building blocks на разумные stock molecules

Практически это уже полноценная planning-задача, где модель должна:

- разбить target на крупные фрагменты
- спланировать последовательность disconnection steps
- удержать глобальную согласованность route
- не потерять branch handling по дороге

Минимальная рабочая форма ответа для такого кейса выглядит так:

```json
{
  "route_depth": 4,
  "reaction_steps": 4,
  "branching_reaction_nodes": 1,
  "terminal_molecules": ["..."],
  "reference_route": {
    "type": "mol",
    "children": [
      {
        "type": "reaction",
        "children": ["..."]
      }
    ]
  }
}
```

Это уже соответствует вопросу не “какой один precursor возможен”, а “может ли модель построить целостный route-level plan до target”.

### Итог по Level C

Это сейчас самый сильный и самый близкий к paper-замыслу уровень.

Он хорошо покрывает:

- `route_depth`
- `branch_handling`
- `convergence`
- `reference_route_planning`
- `route_design`

## Что мы скачали, но не встроили напрямую

Часть источников есть локально, но в benchmark напрямую не пошли:

- `Lowe USPTO`
- `ORD`
- `USPTO-LLM`

### Почему не пошли напрямую

- это скорее большие `reaction corpora`, чем готовые `benchmark tasks`
- им нужна дополнительная фильтрация
- их логичнее использовать как `expansion material`, а не как `immediate benchmark rows`

Также остались `blocked/commercial`:

- `RMechDB`
- `Pistachio`
- `Reaxys`

## Как мы встроили это в репозиторий

Мы сделали единый `row schema`. У каждой записи есть:

- `record_id`
- `level`
- `source_id`
- `source_split`
- `source_license`
- `task_family`
- `task_subtype`
- `difficulty`
- `coverage_tags`
- `input_text`
- `input`
- `gold`
- `metadata`

Это важно, потому что раньше старый benchmark смешивал смысл задачи и формат ответа. Теперь benchmark rows описывают именно задачу, источник и покрытие.

## Как делили по сложности

Сложность задаётся полем `difficulty`.

Общая логика такая.

### Level A

- `easy`: `classification-style rows`
- `medium`: `reagent/procedure role extraction`
- `hard`: `reaction-center / more mechanistically loaded cases`

### Level B

- `easy`: простые `precursor tasks`
- `medium`: более содержательные `one-step disconnections`
- `hard`: более сложные `structurally constrained cases`

### Level C

- `easy/medium/hard` вычисляется из `route structure`, но практически meaningful buckets сейчас в основном `medium/hard`
- для paper eval это отражено явно: в `C` используется в основном `medium/hard`

Особенно важно, что для `eval subsets` мы делали балансировку:

- `A` — по `source` и `subtype`
- `B` — по `source` и `difficulty`
- `C` — по `route source` и `difficulty`

## Что покрывает benchmark по смыслу

Если смотреть содержательно:

### Level A

“что происходит в реакции?”

- `identify reaction center`
- `classify transformation`
- `infer mechanistic class`
- `extract reagent/condition roles`

### Level B

“из чего это собрать за один шаг назад?”

- `propose immediate precursors`
- `local disconnection logic`
- `basic constraints`

### Level C

“как синтезировать это целиком?”

- `full route proposal`
- `route depth`
- `branching`
- `convergence`
- `route-level planning`

То есть benchmark сейчас реально соответствует вашей лестнице:

- `reaction understanding`
- `one-step retrosynthesis`
- `full synthesis planning`

## Что получилось хорошо

Самые сильные стороны текущей конструкции:

- benchmark больше не смешивает всё подряд
- есть чёткая `A/B/C capability ladder`
- есть большие `pools` для агентов
- есть маленькие deterministic `eval subsets` для paper
- источники и provenance зафиксированы явно
- benchmark встроен в текущий runtime и уже `smoke-tested`

## Что пока неполно

Есть и слабые места.

### Level A

- ещё слабоват на `selectivity` и `stereochemical explanation`
- много хорошего coverage, но не идеальный `mechanistic benchmark`

### Level B

- `operationally` сильный
- но всё ещё ближе к `precursor prediction`, чем к глубокой `retrosynthetic justification`

### Level C

- концептуально сильный
- но он дорогой по времени для `baseline runs`

Плюс есть методологический риск:

- некоторые `eval subsets` пока формально “balanced”, но `fallback logic` может слегка размывать идеальные квоты
- это надо будет дочистить, если хочется совсем жёсткий `paper artifact`

## Как этим пользоваться

Если нужна массовая раздача агентам:

- использовать `level_a.jsonl`, `level_b.jsonl`, `level_c.jsonl`

Если нужен `paper evaluation`:

- использовать `level_a_eval.jsonl`, `level_b_eval.jsonl`, `level_c_eval.jsonl`

Если нужен `overview`:

- читать `docs/benchmark_construction.md`
- короткое описание уровней — `docs/benchmark_ladder.md`

## Короткий итог

Мы собрали новый `benchmark v3-style ladder` из нескольких внешних источников и внутреннего пилота, разложили его на `A/B/C`, нормализовали в единую схему, сделали большие `agent pools` и компактные `paper eval subsets`.

- `A` строится вокруг `reaction understanding`
- `B` — вокруг `one-step retrosynthesis`
- `C` — вокруг `route planning`

На сегодня это уже хороший operational benchmark и вполне осмысленная основа для paper, хотя `A` и `B` ещё можно доусилить по `richness of reasoning`.
