# wandb: что занимает место и что можно удалить

Считано только чтением (аккаунт `danrd`), ничего не удалялось. Сырые данные: `sizes.jsonl` (по запускам),
`artifacts.json` (артефакты проектов ARC-inference и llm-run).

Всего по запускам: файлы 0,3 ГБ + артефакты 2,5 ГБ = около 2,8 ГБ в 762 запусках, 10 проектах.

| Проект | Запусков | Упавших | Размер |
|---|---:|---:|---:|
| ARC-inference | 114 | 3 | 1,48 ГБ |
| llm-run | 65 | 1 | 0,85 ГБ |
| cosmos_qa | 137 | 53 | 0,24 ГБ |
| FB3 | 135 | 41 | 0,13 ГБ |
| ARC_RL_object | 263 | 10 | 0,06 ГБ |
| cosmosQA | 34 | 4 | 0,04 ГБ |
| остальные четыре | 14 | 2 | < 0,01 ГБ |

## Где место

Почти всё (около 2,0 ГБ из 2,8) занимают артефакты типа `checkpoint`: у каждого запуска свой артефакт
`checkpoint-<id>` с тысячами версий (ARC-inference: 6463 версии, 1,21 ГБ; llm-run: 4786 версий, 0,78 ГБ).
Остальное мелочь: наборы `*_task_*` в llm-run (около 40 МБ), события запусков (по 1 МБ).

## Кандидаты на удаление (по убыванию пользы)

1. **Старые версии артефактов `checkpoint-*` в ARC-inference и llm-run, кроме последней версии каждого.**
   Освободит почти все 2,0 ГБ. Сами запуски, графики и метрики остаются.
2. **Упавшие запуски (112):** cosmos_qa 53, FB3 41, ARC_RL_object 10, cosmosQA 4, ARC-inference 3,
   huggingface 2, llm-run 1. Места почти не дают (около 0,1 ГБ), но убирают мусор из списков.
3. **Проекты 2022-2023 годов целиком:** cosmos_qa, cosmosQA, FB3, LLM_ScEx, U.S. Patent Phrase to Phrase
   Matching (около 0,4 ГБ). Только если эти работы больше не нужны.

Не трогать: ARC_RL_object (RL-эксперименты 2025), последние версии чекпоинтов, llm-run за 2026.

## Упавшие запуски по проектам

- cosmos_qa (53): qmxiez8a, 2tsocpsm, 781qp5ka, 2rfylrvj, 260dxh5w, d8c16n8e, 2bz58b6f, 37hokxtd, 1hfzc5k0, 2rt4nape, 16crjrpa, 241hk9ja, vuh4kdb9, r8nnfoib, 2lzbudm9, 1pduauyx, 13b927um, 1w1bfkfl, 21do49bv, gcpeuusq, 2pcvucu9, 2fkoztor, 22yvq1k9, 7a5gw42v, tqfv8d26, 3b476xcs, rytlfmmr, 1ec5gayr, 2lpz0ob3, 1hcjqb7j, 1wic2ymq, c3kxr26u, 1uwl3zf1, 1topi0o8, 3hrluatl, qtyfs1fv, jv2qlswz, 3lv95z8d, 7k4d65gp, 3nqk55c4, 2qggo45i, 3knmshpq, harav3aj, 32yg17k8, 1jyun5bf, 3g3bi22o, 1anpbpzy, 3hnbtv3r, hkgq9cqi, beyyir70, 1drmqvbe, 80bh57i6, 1tf9i5ug
- FB3 (41): gd5d62xy, 22b8m32t, vlgzayi4, 2911begl, 6vnayl1b, 2cxtynvn, 3h5b6fx4, 1nexaw2a, 3nqj0uhg, 3mmm5xd3, 3iaf9mo1, l56hmsmj, 2fli2n6a, s5cs0u5u, 4ap4fn3u, 3jsjpn0d, 2d3raqck, 1oqpxnt0, 1et0r2ay, 13ydnczu, i9auqfcs, 1wog6ijc, 3uxdu1mq, ge7bgx37, 2tcqrv5m, jq2a3uxc, 590599se, 1zu9mdap, 3auwurru, 272uebz4, 3oq1skw7, 38xqf870, 2r098a1v, q6d43xuo, 22x97x6f, cbbhtd6x, 1qsi2042, 2e6u5u7c, 3afk4bkc, 26w0l9je, 1hyqt18e
- ARC_RL_object (10): pk5y862g, kq9ammpv, h8z6v751, 2q16ow04, 8lzq47oa, lo51uwvy, p7p567qp, deie31g1, du2xv6ee, 480ejljb
- cosmosQA (4): 1pd3a60s, 1yt3fk6d, 2ctd3h3n, 23emawhn
- ARC-inference (3): htwt5xvi, iwv85lpv, l18mnjia
- huggingface (2): jf5yozsf, usy8yzd9
- llm-run (1): 7vd1hzdx
